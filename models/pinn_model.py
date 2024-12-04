import torch
import torch.nn as nn
import numpy as np

class SolarPositionLayer(nn.Module):
    def __init__(self):
        super(SolarPositionLayer, self).__init__()
        
    def calculate_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))
    
    def calculate_hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        hour = (time * 24) % 24
        return torch.deg2rad(15 * (hour - 12) + lon)
    
    def calculate_zenith(self, lat, declination, hour_angle):
        """Calculate solar zenith angle"""
        lat_rad = torch.deg2rad(lat)
        return torch.arccos(
            torch.sin(lat_rad) * torch.sin(declination) +
            torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle)
        )
    
    def forward(self, x):
        lat, lon, time = x[:, 0], x[:, 1], x[:, 2]
        declination = self.calculate_declination(time)
        hour_angle = self.calculate_hour_angle(time, lon)
        zenith_angle = self.calculate_zenith(lat, declination, hour_angle)
        return zenith_angle

class AtmosphericTransmissionLayer(nn.Module):
    def __init__(self):
        super(AtmosphericTransmissionLayer, self).__init__()
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        
    def forward(self, x, zenith_angle):
        wavelength = x[:, 7]  # Wavelength is the 8th input
        optical_depth = self.beta * (wavelength/0.5)**(-self.alpha)
        air_mass = 1.0 / torch.cos(zenith_angle).clamp(min=0.001)
        return torch.exp(-optical_depth * air_mass)

class SolarPINN(nn.Module):
    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        self.solar_constant = 1367.0  # W/m²
        
        # Physics-informed layers
        self.solar_position_layer = SolarPositionLayer()
        self.atmospheric_layer = AtmosphericTransmissionLayer()
        
        # Core neural network with SiLU activation
        self.physics_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Calculate solar position and atmospheric transmission
        zenith_angle = self.solar_position_layer(x)
        atm_transmission = self.atmospheric_layer(x, zenith_angle)
        
        # Neural network prediction
        y_pred = torch.sigmoid(self.physics_net(x)) * self.solar_constant
        
        # Apply nighttime constraint
        nighttime_mask = (torch.cos(zenith_angle) <= 0)
        y_pred = torch.where(nighttime_mask, torch.zeros_like(y_pred), y_pred)
        
        return y_pred
    
    def compute_physics_loss(self, x, y_pred):
        # Extract inputs
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = [x[:, i] for i in range(8)]
        
        # Calculate solar position
        zenith_angle = self.solar_position_layer(x)
        cos_zenith = torch.cos(zenith_angle)
        
        # Calculate clear sky irradiance
        atm_transmission = self.atmospheric_layer(x, zenith_angle)
        clear_sky = self.solar_constant * atm_transmission * cos_zenith.clamp(min=0)
        
        # Cloud effect (cubic dependency)
        cloud_transmission = 1.0 - 0.75 * (cloud_cover ** 3)
        
        # Geometric factors for tilted surface
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        geometric_factor = (
            torch.cos(slope_rad) * cos_zenith +
            torch.sin(slope_rad) * torch.sin(zenith_angle) * torch.cos(aspect_rad)
        ).clamp(min=0)
        
        # Calculate expected irradiance
        expected_irradiance = clear_sky * cloud_transmission * geometric_factor * atm
        
        # Physics residual with enhanced gradients
        physics_residual = torch.abs(y_pred - expected_irradiance) + \
                          0.1 * torch.abs(torch.gradient(y_pred, dim=0)[0])
        
        # Ensure predictions stay within physical bounds
        bound_violation = torch.relu(y_pred - self.solar_constant) + torch.relu(-y_pred)
        
        return physics_residual.mean() + 10.0 * bound_violation.mean()

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x_batch)
        
        # Calculate losses
        data_loss = self.mse_loss(y_pred, y_batch)
        physics_loss = self.model.compute_physics_loss(x_batch, y_pred)
        
        # Combine losses with adaptive weights
        physics_weight = torch.exp(-physics_loss)
        total_loss = data_loss + physics_weight * physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), data_loss.item(), physics_loss.item()