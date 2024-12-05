import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Constants from the summary table
        self.solar_constant = 1367  # W/m²
        self.extinction_coefficient = 0.1
        self.h = 6.626e-34  # Planck constant
        self.c = 3.0e8     # Speed of light
        self.k = 1.381e-23 # Boltzmann constant
        self.T = 5778      # Solar surface temperature (K)
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),  # Input: latitude, longitude, time, day_of_year, slope, azimuth
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output: predicted irradiance
        )
    
    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle"""
        return 23.45 * torch.sin(2 * np.pi * (day_of_year - 81) / 365)
    
    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using the provided formula"""
        zenith_angle_deg = torch.rad2deg(zenith_angle)
        return 1.0 / (torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
    
    def calculate_atmospheric_transmission(self, air_mass):
        """Calculate atmospheric transmission using Beer-Lambert law"""
        return torch.exp(-self.extinction_coefficient * air_mass)
    
    def calculate_cos_incidence(self, zenith_angle, slope, sun_azimuth, panel_azimuth):
        """Calculate cosine of incidence angle for the panel"""
        return (torch.cos(zenith_angle) * torch.cos(slope) +
                torch.sin(zenith_angle) * torch.sin(slope) * 
                torch.cos(sun_azimuth - panel_azimuth))
    
    def calculate_top_of_atmosphere_irradiance(self, latitude, declination, hour_angle):
        """Calculate irradiance at top of atmosphere"""
        cos_theta = (torch.sin(torch.deg2rad(latitude)) * torch.sin(torch.deg2rad(declination)) +
                    torch.cos(torch.deg2rad(latitude)) * torch.cos(torch.deg2rad(declination)) * 
                    torch.cos(hour_angle))
        return self.solar_constant * torch.maximum(cos_theta, torch.tensor(0.0))
    
    def nighttime_loss(self, time, latitude, declination):
        """Calculate sunrise/sunset times and enforce nighttime conditions"""
        # Convert inputs to tensors if they aren't already
        batch_size = time.shape[0] if isinstance(time, torch.Tensor) else len(time)
        
        # Create dummy values for additional parameters
        longitude = torch.zeros_like(time)  # Assuming longitude doesn't affect nighttime calculation
        slope = torch.zeros_like(time)      # Assuming flat surface
        panel_azimuth = torch.zeros_like(time)  # Assuming south-facing
        
        hour_angle_sunrise = torch.arccos(-torch.tan(torch.deg2rad(latitude)) * 
                                        torch.tan(torch.deg2rad(declination)))
        sunrise = 12 - torch.rad2deg(hour_angle_sunrise) / 15
        sunset = 12 + torch.rad2deg(hour_angle_sunrise) / 15
        
        # Get day_of_year from declination using inverse of declination calculation
        day_of_year = torch.asin(torch.deg2rad(declination) / torch.deg2rad(torch.tensor(23.45))) * 365 / (2 * np.pi) + 81
        
        return torch.mean(torch.where((time < sunrise) | (time > sunset), 
                                    self.forward(latitude, longitude, time, day_of_year, slope, panel_azimuth), 
                                    torch.tensor(0.0)))
    
    def normalize_inputs(self, x, min_val, max_val):
        """Normalize inputs to [-1, 1] range"""
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def forward(self, latitude, longitude, time, day_of_year, slope=0, panel_azimuth=0):
        """Forward pass through the network with physical constraints"""
        # Convert all inputs to tensors and ensure they have the correct shape
        batch_size = latitude.shape[0] if isinstance(latitude, torch.Tensor) else len(latitude)
        
        def to_tensor(x, default=None):
            if isinstance(x, torch.Tensor):
                return x.reshape(batch_size, 1)
            if default is not None and not isinstance(x, (list, np.ndarray)):
                x = [default] * batch_size
            return torch.tensor(x, dtype=torch.float32).reshape(batch_size, 1)
        
        # Normalize inputs to prevent numerical instability
        inputs = torch.cat([
            to_tensor(self.normalize_inputs(latitude, -90, 90)),
            to_tensor(self.normalize_inputs(longitude, -180, 180)),
            to_tensor(self.normalize_inputs(time, 0, 24)),
            to_tensor(self.normalize_inputs(day_of_year, 0, 365)),
            to_tensor(self.normalize_inputs(slope, -90, 90)),
            to_tensor(self.normalize_inputs(panel_azimuth, -180, 180))
        ], dim=1)
        
        # Get base prediction from neural network
        predicted_irradiance = self.network(inputs)
        
        # Apply physical constraints
        declination = self.calculate_declination(day_of_year)
        hour_angle = (time - 12) * 15  # Convert hour to degree
        
        # Calculate zenith angle
        zenith_angle = torch.arccos(
            torch.sin(torch.deg2rad(latitude)) * torch.sin(torch.deg2rad(declination)) +
            torch.cos(torch.deg2rad(latitude)) * torch.cos(torch.deg2rad(declination)) * 
            torch.cos(torch.deg2rad(hour_angle))
        )
        
        # Apply physical factors
        air_mass = self.calculate_air_mass(zenith_angle)
        transmission = self.calculate_atmospheric_transmission(air_mass)
        cos_incidence = self.calculate_cos_incidence(zenith_angle, slope, hour_angle, panel_azimuth)
        
        # Modify prediction based on physical constraints
        modified_irradiance = predicted_irradiance * transmission * torch.maximum(cos_incidence, torch.tensor(0.0))
        
        return torch.maximum(modified_irradiance, torch.tensor(0.0))  # Ensure non-negative output

    def compute_loss(self, pred, target, latitude, time, day_of_year):
        """Compute total loss including physical constraints with scaled components"""
        eps = 1e-8
        
        # MSE loss between predictions and targets (scaled)
        pred_scaled = pred / self.solar_constant
        target_scaled = target / self.solar_constant
        mse_loss = torch.mean(torch.clamp((pred_scaled - target_scaled) ** 2, min=0.0, max=1.0))
        
        # Physical constraint losses (scaled)
        declination = torch.clamp(self.calculate_declination(day_of_year), min=-23.45, max=23.45)
        night_loss = 0.1 * self.nighttime_loss(time, latitude, declination)  # Reduced weight
        
        # Maximum irradiance constraint (scaled)
        max_constraint = 0.01 * torch.mean(
            torch.clamp(torch.relu(pred / self.solar_constant - 1.0), max=1.0)
        )
        
        # Minimum irradiance constraint (scaled)
        min_constraint = 0.01 * torch.mean(
            torch.clamp(torch.relu(-pred_scaled), max=1.0)
        )
        
        # Combine losses with balanced weights
        total_loss = (
            mse_loss +
            night_loss +
            max_constraint +
            min_constraint
        )
        
        # Add small epsilon to prevent complete zero loss
        total_loss = total_loss + eps
        
        return torch.where(
            torch.isnan(total_loss) | (total_loss > 1e3),
            torch.tensor(10.0, dtype=total_loss.dtype),
            total_loss
        )
