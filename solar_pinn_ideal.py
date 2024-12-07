import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics-informed constraints for solar irradiance prediction."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights with consideration for solar physics
        self.physics_weights = nn.Parameter(torch.ones(out_features) * 0.5)

    def forward(self, x):
        out = self.linear(x)
        # Ensure positive output through physics-based transformation
        out = out * torch.sigmoid(self.physics_weights)
        return out

class SolarPINN(nn.Module):
    def __init__(self, input_dim=6):  # latitude, longitude, time, day_of_year, slope, aspect
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
    
    def setup_physical_constants(self):
        """Set up essential physical constants for ideal conditions."""
        self.solar_constant = 1367.0  # Solar constant at top of atmosphere (W/m²)
        self.atmospheric_extinction = 0.1  # Idealized clear-sky extinction coefficient

    def setup_network(self, input_dim):
        """Setup neural network architecture with batch normalization."""
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            PhysicsInformedLayer(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 1)
        )

    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle using exact formulation.
        
        Args:
            day_of_year: Day of year (1-365)
            
        Returns:
            Solar declination angle in radians
        """
        return torch.deg2rad(23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365))

    def calculate_hour_angle(self, time):
        """Calculate hour angle."""
        return torch.deg2rad(15 * (time - 12))  # Convert hour to radians

    def calculate_zenith_angle(self, lat, declination, hour_angle):
        """Calculate solar zenith angle."""
        lat_rad = torch.deg2rad(lat)
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) + 
                     torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle))
        return torch.arccos(torch.clamp(cos_zenith, -1, 1))

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten & Young formula."""
        zenith_deg = torch.rad2deg(zenith_angle)
        denominator = torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_deg) ** -1.6364
        return torch.where(zenith_deg < 90, 1 / denominator, float('inf'))

    def calculate_surface_orientation_factor(self, zenith_angle, slope, sun_azimuth, aspect):
        """Calculate surface orientation factor."""
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        sun_azimuth_rad = torch.deg2rad(sun_azimuth)
        
        return (torch.cos(slope_rad) * torch.cos(zenith_angle) + 
                torch.sin(slope_rad) * torch.sin(zenith_angle) * 
                torch.cos(sun_azimuth_rad - aspect_rad))

    def boundary_conditions(self, lat, day_of_year):
        """Calculate sunrise, sunset, and daylight duration using exact formulations.
        
        Args:
            lat: Latitude of location (in degrees)
            day_of_year: Day of year (1-365)
            
        Returns:
            tuple: (sunrise time, sunset time) in hours (local solar time)
        """
        # Calculate solar declination
        declination = self.calculate_declination(day_of_year)
        
        # Convert latitude to radians
        lat_rad = torch.deg2rad(lat)
        
        # Calculate hour angle at sunrise/sunset
        # cos(h) = -tan(φ)·tan(δ)
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(declination)
        
        # Handle edge cases (polar days/nights)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        
        # Calculate hour angle in radians
        hour_angle = torch.arccos(cos_hour_angle)
        
        # Convert to degrees for time calculations
        hour_angle_deg = torch.rad2deg(hour_angle)
        
        # Calculate sunrise and sunset times
        # Sunrise = 12 - h/15, Sunset = 12 + h/15
        sunrise = 12 - hour_angle_deg / 15
        sunset = 12 + hour_angle_deg / 15
        
        # Handle special cases
        is_polar_day = cos_hour_angle < -1
        is_polar_night = cos_hour_angle > 1
        
        # During polar day, sun never sets (24h daylight)
        sunrise = torch.where(is_polar_day, torch.zeros_like(sunrise), sunrise)
        sunset = torch.where(is_polar_day, torch.full_like(sunset, 24), sunset)
        
        # During polar night, sun never rises (24h darkness)
        sunrise = torch.where(is_polar_night, torch.full_like(sunrise, float('inf')), sunrise)
        sunset = torch.where(is_polar_night, torch.full_like(sunset, float('inf')), sunset)
        
        return sunrise, sunset

    def calculate_daylight_duration(self, lat, day_of_year):
        """Calculate daylight duration using exact formulation.
        
        Args:
            lat: Latitude of location (in degrees)
            day_of_year: Day of year (1-365)
            
        Returns:
            Daylight duration in hours
        """
        sunrise, sunset = self.boundary_conditions(lat, day_of_year)
        return sunset - sunrise

    def forward(self, x):
        """Forward pass with essential physics constraints using normalized inputs."""
        # Extract normalized input components
        lat_norm, lon_norm, time_norm, day_norm, slope_norm, aspect_norm = x.split(1, dim=1)
        
        # Denormalize inputs for physics calculations
        lat = lat_norm * 90
        lon = lon_norm * 180
        time = time_norm * 24  # Now represents only hours (0-24)
        day_of_year = day_norm * 364 + 1  # Scale to [1, 365]
        slope = slope_norm * 180
        aspect = aspect_norm * 360
        
        hour_of_day = time  # Time now directly represents hours
        
        # Calculate solar position
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        zenith_angle = self.calculate_zenith_angle(lat, declination, hour_angle)
        
        # Calculate atmospheric effects
        air_mass = self.calculate_air_mass(zenith_angle)
        atmospheric_transmission = torch.exp(-self.atmospheric_extinction * air_mass)
        
        # Calculate surface orientation
        sun_azimuth = torch.rad2deg(hour_angle)  # Simplified sun azimuth
        surface_factor = self.calculate_surface_orientation_factor(zenith_angle, slope, sun_azimuth, aspect)
        
        # Neural network prediction (normalized)
        prediction = self.physics_net(x)
        
        # Re-dimensionalize the prediction and apply physical constraints
        prediction = prediction * self.solar_constant  # Convert normalized output back to W/m²
        constrained_prediction = (prediction * 
                               atmospheric_transmission * 
                               surface_factor)
        
        # Apply day/night boundary conditions
        sunrise, sunset = self.boundary_conditions(lat, day_of_year)
        is_daytime = (hour_of_day >= sunrise) & (hour_of_day <= sunset)
        
        final_prediction = torch.where(is_daytime, 
                                    torch.clamp(constrained_prediction, min=0.0),
                                    torch.zeros_like(constrained_prediction))
        
        # Return normalized prediction for training
        return final_prediction / self.solar_constant

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, x_data, y_data, physics_weight=0.15):
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x_data)
        
        # Calculate data loss
        data_loss = F.mse_loss(y_pred, y_data)
        
        # Extract time components for boundary conditions
        lat = x_data[:, 0] * 90
        time = x_data[:, 2] * 24
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Calculate physics loss
        sunrise, sunset = self.model.boundary_conditions(lat, day_of_year)
        night_mask = (hour_of_day < sunrise) | (hour_of_day > sunset)
        physics_loss = torch.mean(y_pred[night_mask] ** 2)
        
        # Combined loss with retain_graph
        total_loss = data_loss + physics_weight * physics_loss
        
        # Backward pass with retain_graph
        total_loss.backward(retain_graph=True)
        
        # Optimizer step
        self.optimizer.step()
        
        # Clean up graphs
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
        
        return total_loss.item()
