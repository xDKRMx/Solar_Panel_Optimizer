import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics-informed constraints for solar irradiance prediction."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights with Kaiming initialization for ReLU
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        self.linear.bias.data.fill_(0.1)
        # Initialize physics weights with small positive values for stability
        self.physics_weights = nn.Parameter(0.1 * torch.ones(out_features))

    def forward(self, x):
        out = self.linear(x)
        # Apply ReLU for positive outputs and scale with physics weights
        out = F.relu(out) * self.physics_weights
        return out

class SolarPINN(nn.Module):
    def __init__(self, input_dim=5):  # latitude, longitude, time, slope, aspect
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
        
    def normalize_inputs(self, x):
        """Normalize each input dimension."""
        lat = x[:, 0] / 90.0  # -1 to 1
        lon = x[:, 1] / 180.0  # -1 to 1
        time = x[:, 2] / 24.0  # 0 to 1
        slope = x[:, 3] / 90.0  # 0 to 1
        aspect = x[:, 4] / 360.0  # 0 to 1
        return torch.stack([lat, lon, time, slope, aspect], dim=1)
        
    def denormalize_output(self, y):
        """Scale back to physical units (W/m²)."""
        return y * self.solar_constant
    
    def setup_physical_constants(self):
        """Set up essential physical constants for ideal conditions."""
        self.solar_constant = 1367.0  # Solar constant at top of atmosphere (W/m²)
        self.atmospheric_extinction = 0.1  # Idealized clear-sky extinction coefficient

    def setup_network(self, input_dim):
        """Setup neural network architecture with batch normalization."""
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PhysicsInformedLayer(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PhysicsInformedLayer(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PhysicsInformedLayer(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PhysicsInformedLayer(128, 1),
            nn.Sigmoid()  # Ensure output is between 0 and 1 for normalization
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
        """Forward pass with essential physics constraints and normalization."""
        # Extract input components
        lat, lon, time, slope, aspect = x.split(1, dim=1)
        
        # Normalize inputs
        x_normalized = self.normalize_inputs(x)
        
        # Neural network prediction (0-1 range)
        prediction = self.physics_net(x_normalized)
        
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
        
        # Apply physical constraints and denormalize
        constrained_prediction = self.denormalize_output(prediction) * atmospheric_transmission * surface_factor
        
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
            
            # Calculate data loss with L1 component for better stability
            mse_loss = F.mse_loss(y_pred, y_data)
            l1_loss = F.l1_loss(y_pred, y_data)
            data_loss = 0.8 * mse_loss + 0.2 * l1_loss
            
            # Extract time components for boundary conditions
            lat = x_data[:, 0]
            time = x_data[:, 2]
            day_of_year = torch.floor(time / 24 * 365)
            hour_of_day = time % 24
            
            # Calculate physics loss
            sunrise, sunset = self.model.boundary_conditions(lat, day_of_year)
            night_mask = (hour_of_day < sunrise) | (hour_of_day > sunset)
            physics_loss = torch.mean(y_pred[night_mask] ** 2)  # Should be zero at night
            
            # Combined loss with adaptive weighting
            physics_weight = min(0.5, 0.1 * (1 + torch.exp(-data_loss)).item())
            total_loss = data_loss + physics_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            return total_loss.item()
        except Exception as e:
            print(f"Training error: {str(e)}")
            return float('inf')
