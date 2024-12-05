import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Essential physical constants for ideal conditions
        self.solar_constant = 1367.0  # W/m² (solar constant at top of atmosphere)
        
        # Neural network for irradiance prediction
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Input: latitude, time, day_of_year, slope
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: predicted irradiance
        )
    
    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle (δ)
        δ = 23.45 * sin(2π(d-81)/365)
        """
        return 23.45 * torch.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    def calculate_hour_angle(self, time):
        """Calculate hour angle (ω)
        ω = (t - 12) * 15
        """
        return (time - 12) * 15  # Convert hour to degrees
    
    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate solar zenith angle"""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        hour_rad = torch.deg2rad(hour_angle)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(decl_rad) + 
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_rad))
        return torch.arccos(torch.clamp(cos_zenith, -1, 1))
    
    def calculate_sunrise_sunset(self, latitude, declination):
        """Calculate sunrise and sunset times using the sunrise equation"""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(decl_rad)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        
        hour_angle = torch.arccos(cos_hour_angle)
        sunrise = 12 - (torch.rad2deg(hour_angle) / 15)
        sunset = 12 + (torch.rad2deg(hour_angle) / 15)
        
        return sunrise, sunset
    
    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten & Young formula"""
        zenith_deg = torch.rad2deg(zenith_angle)
        denominator = torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364)
        return torch.where(zenith_deg < 90, 1 / denominator, float('inf'))
    
    def calculate_surface_factor(self, zenith_angle, slope):
        """Calculate surface orientation factor for ideal conditions (south-facing)
        surface_factor = cos(slope)*cos(zenith) + sin(slope)*sin(zenith)
        """
        if not isinstance(slope, torch.Tensor):
            slope = torch.full_like(zenith_angle, float(slope))
        slope_rad = torch.deg2rad(slope)
        
        surface_factor = (torch.cos(slope_rad) * torch.cos(zenith_angle) + 
                         torch.sin(slope_rad) * torch.sin(zenith_angle))
        return torch.clamp(surface_factor, 0, 1)
    
    def calculate_toa_irradiance(self, latitude, declination, hour_angle):
        """Calculate top of atmosphere irradiance
        I = S₀ * (sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω))
        """
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        hour_rad = torch.deg2rad(hour_angle)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(decl_rad) + 
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_rad))
        
        return self.solar_constant * torch.clamp(cos_zenith, 0, 1)
    
    def forward(self, latitude, time, day_of_year, slope=0):
        """Forward pass with essential physics constraints for ideal conditions"""
        # Neural network prediction
        inputs = torch.cat([
            latitude.reshape(-1, 1),
            time.reshape(-1, 1),
            day_of_year.reshape(-1, 1),
            torch.full_like(latitude.reshape(-1, 1), float(slope))
        ], dim=1)
        
        predicted_irradiance = self.network(inputs)
        
        # Calculate solar position parameters
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        zenith_angle = self.calculate_zenith_angle(latitude, declination, hour_angle)
        
        # Calculate physical constraints
        toa_irradiance = self.calculate_toa_irradiance(latitude, declination, hour_angle)
        air_mass = self.calculate_air_mass(zenith_angle)
        surface_factor = self.calculate_surface_factor(zenith_angle, slope)
        
        # Simple clear sky model with only air mass
        atmospheric_transmission = torch.exp(-0.1 * air_mass)  # Simple Beer-Lambert law
        constrained_irradiance = predicted_irradiance * atmospheric_transmission * surface_factor
        
        # Apply day/night constraint
        sunrise, sunset = self.calculate_sunrise_sunset(latitude, declination)
        day_mask = (time >= sunrise) & (time <= sunset)
        constrained_irradiance = torch.where(day_mask.reshape(-1, 1), 
                                           constrained_irradiance, 
                                           torch.zeros_like(constrained_irradiance))
        
        # Ensure output is within physical bounds
        return torch.clamp(constrained_irradiance, 0, toa_irradiance.reshape(-1, 1))
    
    def compute_loss(self, pred, target, latitude, time, day_of_year):
        """Simple physics-informed loss function for ideal conditions"""
        # MSE loss between prediction and target
        mse_loss = torch.mean((pred - target) ** 2)
        
        # Physics-based loss components
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        toa_irradiance = self.calculate_toa_irradiance(latitude, declination, hour_angle)
        
        # Physics violations penalties
        toa_violation = torch.mean(torch.relu(pred - toa_irradiance.reshape(-1, 1)))
        sunrise, sunset = self.calculate_sunrise_sunset(latitude, declination)
        night_violation = torch.mean(torch.relu(pred * 
                                              (~((time >= sunrise) & 
                                                 (time <= sunset))).float().reshape(-1, 1)))
        
        # Combined loss with physics constraints
        total_loss = mse_loss + 0.1 * toa_violation + 0.1 * night_violation
        return total_loss
