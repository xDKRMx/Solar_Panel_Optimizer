import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics constraints."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.physics_weights = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        out = self.linear(x)
        out = out * torch.sigmoid(self.physics_weights)  # Physics-based transformation
        return out


class SolarPINN(nn.Module):
    def __init__(self, input_dim=4):  # latitude, time, day_of_year, slope
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
    
    def setup_physical_constants(self):
        """Set up physical constants and parameters."""
        # Universal constants
        self.solar_constant = 1367.0  # W/m²
        
        # Ideal atmospheric parameters
        self.atmospheric_extinction = 0.1  # Idealized extinction coefficient

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
        return 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)

    def calculate_hour_angle(self, time):
        """Calculate hour angle."""
        return (time - 12) * 15  # Convert hour to degrees

    def calculate_zenith_angle(self, lat, declination, hour_angle):
        """Calculate solar zenith angle."""
        lat_rad = torch.deg2rad(lat)
        decl_rad = torch.deg2rad(declination)
        hour_rad = torch.deg2rad(hour_angle)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(decl_rad) + 
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_rad))
        return torch.arccos(torch.clamp(cos_zenith, -1, 1))

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten & Young formula."""
        zenith_deg = torch.rad2deg(zenith_angle)
        denominator = torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364)
        return torch.where(zenith_deg < 90, 1 / denominator, float('inf'))

    def calculate_surface_factor(self, zenith_angle, slope):
        """Calculate surface orientation factor for ideal conditions (south-facing)."""
        slope_rad = torch.deg2rad(slope)
        surface_factor = (torch.cos(slope_rad) * torch.cos(zenith_angle) + 
                        torch.sin(slope_rad) * torch.sin(zenith_angle))
        return torch.clamp(surface_factor, 0, 1)

    def calculate_sunrise_sunset(self, latitude, declination):
        """Calculate sunrise and sunset times."""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(decl_rad)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        
        hour_angle = torch.arccos(cos_hour_angle)
        sunrise = 12 - (torch.rad2deg(hour_angle) / 15)
        sunset = 12 + (torch.rad2deg(hour_angle) / 15)
        
        return sunrise, sunset

    def forward(self, x):
        """Forward pass with physics constraints."""
        # Extract individual components
        latitude = x[:, 0]
        time = x[:, 1]
        day_of_year = x[:, 2]
        slope = x[:, 3]
        
        # Neural network prediction
        prediction = self.physics_net(x)
        
        # Calculate physical parameters
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        zenith_angle = self.calculate_zenith_angle(latitude, declination, hour_angle)
        
        # Apply physical constraints
        air_mass = self.calculate_air_mass(zenith_angle)
        surface_factor = self.calculate_surface_factor(zenith_angle, slope)
        
        # Calculate atmospheric transmission
        atmospheric_transmission = torch.exp(-self.atmospheric_extinction * air_mass)
        
        # Apply physical constraints
        constrained_prediction = prediction * atmospheric_transmission * surface_factor
        
        # Apply day/night constraint
        sunrise, sunset = self.calculate_sunrise_sunset(latitude, declination)
        day_mask = (time >= sunrise) & (time <= sunset)
        final_prediction = torch.where(day_mask.reshape(-1, 1), 
                                     constrained_prediction, 
                                     torch.zeros_like(constrained_prediction))
        
        return final_prediction


class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model(x_data)

        # Calculate losses
        data_loss = F.mse_loss(y_pred, y_data)
        
        # Add physics-informed loss (day/night boundary conditions)
        lat, time = x_data[:, 0], x_data[:, 1]
        day_of_year = x_data[:, 2]
        
        declination = self.model.calculate_declination(day_of_year)
        sunrise, sunset = self.model.calculate_sunrise_sunset(lat, declination)
        night_mask = (time < sunrise) | (time > sunset)
        physics_loss = torch.mean(y_pred[night_mask] ** 2)  # Should be zero at night
        
        # Combined loss
        total_loss = data_loss + 0.1 * physics_loss

        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
