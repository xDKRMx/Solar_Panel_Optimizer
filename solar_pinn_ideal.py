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
    def __init__(self, input_dim=5):  # latitude, longitude, time, slope, aspect
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
    
    def setup_physical_constants(self):
        """Set up essential physical constants for ideal conditions."""
        self.solar_constant = 1367.0  # Solar constant at top of atmosphere (W/m²)
        self.atmospheric_extinction = 0.1  # Idealized clear-sky extinction coefficient

    def setup_network(self, input_dim):
        """Setup neural network architecture."""
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 256),
            nn.Tanh(),
            PhysicsInformedLayer(256, 128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 1)
        )

    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle."""
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
        """Calculate sunrise and sunset times."""
        declination = self.calculate_declination(day_of_year)
        lat_rad = torch.deg2rad(lat)
        
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(declination)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        
        hour_angle = torch.arccos(cos_hour_angle)
        sunrise = 12 - torch.rad2deg(hour_angle) / 15
        sunset = 12 + torch.rad2deg(hour_angle) / 15
        
        return sunrise, sunset

    def forward(self, x):
        """Forward pass with essential physics constraints."""
        # Extract input components
        lat, lon, time, slope, aspect = x.split(1, dim=1)
        
        # Calculate day of year from time (assuming time includes day information)
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
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
        
        # Neural network prediction
        prediction = self.physics_net(x)
        
        # Apply physical constraints
        constrained_prediction = (prediction * 
                                atmospheric_transmission * 
                                surface_factor * 
                                self.solar_constant)
        
        # Apply day/night boundary conditions
        sunrise, sunset = self.boundary_conditions(lat, day_of_year)
        is_daytime = (hour_of_day >= sunrise) & (hour_of_day <= sunset)
        
        return torch.where(is_daytime, 
                         torch.clamp(constrained_prediction, min=0.0),
                         torch.zeros_like(constrained_prediction))

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, x_data, y_data):
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        try:
            # Forward pass with gradient computation
            y_pred = self.model(x_data)
            
            # Calculate data loss
            data_loss = F.mse_loss(y_pred, y_data)
            
            # Extract time components for boundary conditions
            lat = x_data[:, 0]
            time = x_data[:, 2]
            day_of_year = torch.floor(time / 24 * 365)
            hour_of_day = time % 24
            
            # Calculate physics loss with retain_graph
            sunrise, sunset = self.model.boundary_conditions(lat, day_of_year)
            night_mask = (hour_of_day < sunrise) | (hour_of_day > sunset)
            physics_loss = torch.mean(y_pred[night_mask] ** 2)  # Should be zero at night
            
            # Combined loss
            total_loss = data_loss + 0.1 * physics_loss
            
            # Backward pass with retain_graph=True for the first backward
            total_loss.backward(retain_graph=True)
            
            # Optimizer step
            self.optimizer.step()
            
            # Clear computation graph
            self.optimizer.zero_grad()
            
            return total_loss.item()
        except Exception as e:
            print(f"Training error: {str(e)}")
            return float('inf')
