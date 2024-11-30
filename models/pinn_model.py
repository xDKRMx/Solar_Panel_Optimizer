import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        # Separate networks for irradiance and efficiency
        self.irradiance_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure non-negative output
        )
        
        self.efficiency_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # For efficiency scaling
        )
        
        self.solar_constant = 1367.0  # W/m²
        self.ref_wavelength = 0.5  # μm, reference wavelength
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        self.cloud_alpha = 0.85  # Cloud transmission parameter
        self.ref_temp = 25.0  # Reference temperature (°C)
        self.temp_coeff = 0.004  # Temperature coefficient (/°C)
        
    def forward(self, x):
        # Predict raw irradiance (0 to solar_constant)
        irradiance = self.irradiance_net(x) * self.solar_constant
        
        # Predict efficiency (15-25%)
        efficiency = 0.15 + self.efficiency_net(x) * 0.10
        
        return irradiance, efficiency
    
    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))
    
    def hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        hour = (time % 24).clamp(0, 24)
        return torch.deg2rad(15 * (hour - 12) + lon)
    
    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ) and zenith angle"""
        lat_rad = torch.deg2rad(lat)
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        
        declination = self.solar_declination(time)
        hour_angle = self.hour_angle(time, lon)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) +
                     torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle))
        
        cos_theta = (torch.sin(lat_rad) * torch.sin(declination) * torch.cos(slope_rad) -
                    torch.cos(lat_rad) * torch.sin(declination) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.cos(slope_rad) +
                    torch.sin(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(declination) * torch.sin(hour_angle) * torch.sin(slope_rad) * torch.sin(aspect_rad))
        
        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)
    
    def calculate_optical_depth(self, wavelength, air_mass, altitude=0, cloud_cover=0):
        """Calculate advanced multi-wavelength optical depth with cloud and altitude effects"""
        base_depth = self.beta * (wavelength / self.ref_wavelength) ** (-self.alpha)
        altitude_factor = torch.exp(-altitude / 7.4)
        cloud_factor = 1.0 + (cloud_cover ** 2) * (1.0 - 0.5 * altitude_factor)
        air_mass_factor = torch.exp(-base_depth * air_mass * altitude_factor * cloud_factor)
        cloud_scatter = 0.2 * cloud_cover * (wavelength / self.ref_wavelength) ** (-0.75)
        return base_depth * air_mass_factor + cloud_scatter
    
    def physics_loss(self, x, irradiance_pred, efficiency_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))
        
        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)
        
        # Strong nighttime penalty (when sun is below horizon)
        nighttime_condition = (cos_zenith <= 0.001)
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(irradiance_pred) * 100.0,
            torch.zeros_like(irradiance_pred)
        )
        
        # Air mass ratio calculation using Kasten and Young's formula
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))
        
        # Calculate optical depth with enhanced cloud physics
        optical_depth = self.calculate_optical_depth(wavelength, air_mass, altitude=0, cloud_cover=cloud_cover)
        
        # Enhanced cloud transmission model
        base_transmission = 1.0 - self.cloud_alpha * (cloud_cover ** 2)
        diffuse_factor = 0.3 * cloud_cover * (1.0 - cos_zenith)
        cloud_transmission = base_transmission + diffuse_factor
        
        # Calculate theoretical irradiance components
        direct_irradiance = (self.solar_constant * 
                           torch.exp(-optical_depth * air_mass) * 
                           cos_theta * 
                           cloud_transmission)
        
        diffuse_irradiance = self.solar_constant * 0.3 * (1.0 - cloud_transmission) * cos_zenith
        
        ground_albedo = 0.2
        reflected_irradiance = ground_albedo * direct_irradiance * (1.0 - cos_theta) * 0.5
        
        # Total theoretical irradiance
        theoretical_irradiance = torch.where(
            cos_zenith > 0,
            direct_irradiance + diffuse_irradiance + reflected_irradiance,
            torch.zeros_like(direct_irradiance)
        )
        
        # Irradiance physics residuals
        irradiance_residual = irradiance_pred - theoretical_irradiance
        boundary_residual = torch.relu(-irradiance_pred)
        max_irradiance_residual = torch.relu(irradiance_pred - self.solar_constant)
        
        # Efficiency constraints
        efficiency_range_penalty = (
            torch.mean(torch.relu(-efficiency_pred + 0.15)) * 1000.0 +  # Strong penalty below 15%
            torch.mean(torch.relu(efficiency_pred - 0.25)) * 1000.0     # Strong penalty above 25%
        )
        
        # Combine all losses
        total_loss = (
            torch.mean(irradiance_residual**2) * 0.4 +
            torch.mean(boundary_residual + max_irradiance_residual) * 0.2 +
            efficiency_range_penalty * 0.4 +
            torch.mean(nighttime_penalty)
        )
        
        return total_loss

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, x_data, y_irradiance, y_efficiency):
        self.optimizer.zero_grad()
        
        # Forward pass
        irradiance_pred, efficiency_pred = self.model(x_data)
        
        # Compute losses
        irradiance_mse = self.mse_loss(irradiance_pred, y_irradiance)
        efficiency_mse = self.mse_loss(efficiency_pred, y_efficiency)
        physics_loss = self.model.physics_loss(x_data, irradiance_pred, efficiency_pred)
        
        # Total loss
        total_loss = irradiance_mse + efficiency_mse + physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), irradiance_mse.item(), efficiency_mse.item(), physics_loss.item()