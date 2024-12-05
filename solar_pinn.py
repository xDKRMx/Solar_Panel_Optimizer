import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Essential physical constants
        self.solar_constant = 1367  # W/m²
        self.extinction_coefficient = 0.1  # Simplified atmospheric extinction coefficient
        
        # Simplified neural network architecture
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),  # Input: latitude, longitude, time, day_of_year, slope, azimuth
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: predicted irradiance
        )
    
    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle"""
        return 23.45 * torch.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    def calculate_hour_angle(self, time):
        """Calculate hour angle from local time"""
        return (time - 12) * 15  # Convert hour to degree
    
    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate solar zenith angle"""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        hour_rad = torch.deg2rad(hour_angle)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(decl_rad) +
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_rad))
        return torch.arccos(torch.clamp(cos_zenith, -1, 1))
    
    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using simplified formula"""
        zenith_angle_deg = torch.rad2deg(zenith_angle)
        return 1.0 / (torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
    
    def calculate_atmospheric_transmission(self, air_mass):
        """Calculate basic atmospheric transmission"""
        return torch.exp(-self.extinction_coefficient * air_mass)
    
    def calculate_surface_factor(self, zenith_angle, slope, azimuth):
        """Calculate simplified surface orientation factor"""
        slope_rad = torch.deg2rad(slope)
        azimuth_rad = torch.deg2rad(azimuth)
        
        surface_factor = (torch.cos(zenith_angle) * torch.cos(slope_rad) +
                         torch.sin(zenith_angle) * torch.sin(slope_rad) * torch.cos(azimuth_rad))
        return torch.clamp(surface_factor, 0, 1)
    
    def normalize_inputs(self, x, min_val, max_val):
        """Normalize inputs to [-1, 1] range"""
        return 2 * (x - min_val) / (max_val - min_val) - 1
    
    def forward(self, latitude, longitude, time, day_of_year, slope=0, panel_azimuth=0):
        """Forward pass with simplified physics constraints"""
        batch_size = latitude.shape[0] if isinstance(latitude, torch.Tensor) else len(latitude)
        
        def to_tensor(x, default=None):
            if isinstance(x, torch.Tensor):
                return x.reshape(batch_size, 1)
            if default is not None and not isinstance(x, (list, np.ndarray)):
                x = [default] * batch_size
            return torch.tensor(x, dtype=torch.float32).reshape(batch_size, 1)
        
        # Normalize inputs
        inputs = torch.cat([
            to_tensor(self.normalize_inputs(latitude, -90, 90)),
            to_tensor(self.normalize_inputs(longitude, -180, 180)),
            to_tensor(self.normalize_inputs(time, 0, 24)),
            to_tensor(self.normalize_inputs(day_of_year, 0, 365)),
            to_tensor(self.normalize_inputs(slope, -90, 90)),
            to_tensor(self.normalize_inputs(panel_azimuth, -180, 180))
        ], dim=1)
        
        # Base prediction
        predicted_irradiance = self.network(inputs)
        
        # Apply physical constraints
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        zenith_angle = self.calculate_zenith_angle(latitude, declination, hour_angle)
        
        # Calculate atmospheric effects
        air_mass = self.calculate_air_mass(zenith_angle)
        transmission = self.calculate_atmospheric_transmission(air_mass)
        
        # Apply surface orientation
        surface_factor = self.calculate_surface_factor(zenith_angle, slope, panel_azimuth)
        
        # Final prediction with physical constraints
        irradiance = predicted_irradiance * transmission * surface_factor
        
        # Ensure output is within physical bounds
        return torch.clamp(irradiance, 0, self.solar_constant)
    
    def compute_loss(self, pred, target, latitude, time, day_of_year):
        """Simplified physics-informed loss function"""
        # MSE loss between predictions and targets
        mse_loss = torch.mean((pred - target) ** 2)
        
        # Physics-based constraints
        solar_constant_loss = torch.mean(torch.relu(pred - self.solar_constant))
        
        # Day/night boundary condition
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        zenith_angle = self.calculate_zenith_angle(latitude, declination, hour_angle)
        night_loss = torch.mean(torch.relu(-torch.cos(zenith_angle)) * pred)
        
        # Combine losses with appropriate weights
        total_loss = mse_loss + 0.1 * solar_constant_loss + 0.1 * night_loss
        
        return total_loss