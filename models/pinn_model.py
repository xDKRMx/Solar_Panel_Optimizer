import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):

    def __init__(self, input_dim=8):  # Updated for cloud_cover and wavelength
        super(SolarPINN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(),
                               nn.Linear(64, 128), nn.Tanh(),
                               nn.Linear(128, 64), nn.Tanh(),
                               nn.Linear(64, 1))
        self.solar_constant = 1367.0  # W/m²
        self.ref_wavelength = 0.5  # μm, reference wavelength for Ångström formula
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        self.cloud_alpha = 0.75  # Empirically derived cloud transmission parameter
        self.ref_temp = 25.0  # Reference temperature (°C)
        self.temp_coeff = 0.004  # Temperature coefficient (/°C)

    def forward(self, x):
        raw_output = self.net(x)
        return torch.sigmoid(raw_output)

    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi *
                                              (284 + day_number) / 365))

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
                     torch.cos(lat_rad) * torch.cos(declination) *
                     torch.cos(hour_angle))

        cos_theta = (
            torch.sin(lat_rad) * torch.sin(declination) * torch.cos(slope_rad)
            - torch.cos(lat_rad) * torch.sin(declination) *
            torch.sin(slope_rad) * torch.cos(aspect_rad) +
            torch.cos(lat_rad) * torch.cos(declination) *
            torch.cos(hour_angle) * torch.cos(slope_rad) +
            torch.sin(lat_rad) * torch.cos(declination) * torch.cos(hour_angle)
            * torch.sin(slope_rad) * torch.cos(aspect_rad) +
            torch.cos(declination) * torch.sin(hour_angle) *
            torch.sin(slope_rad) * torch.sin(aspect_rad))

        return torch.clamp(cos_theta, min=0.0), torch.clamp(
            cos_zenith, min=0.0001)  # Avoid division by zero

    def calculate_optical_depth(self, wavelength):
        """Calculate optical depth using Ångström turbidity formula"""
        # Exact Ångström turbidity formula from solar_physics.py
        return self.beta * (wavelength / self.ref_wavelength)**(-self.alpha)

    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))

        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(
            lat, lon, time, slope, aspect)

        # Direct zero assignment for nighttime (cos_zenith <= 0)
        nighttime_mask = (cos_zenith <= 0)
        y_pred = torch.where(nighttime_mask, torch.zeros_like(y_pred), y_pred)

        # Calculate air mass using Kasten and Young formula
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))

        # Calculate optical depth using simplified Ångström formula
        optical_depth = self.calculate_optical_depth(wavelength)

        # Simple cloud transmission model aligned with solar_physics.py
        cloud_transmission = 1.0 - self.cloud_alpha * (cloud_cover**3)

        # Simplified irradiance calculations
        direct_irradiance = (self.solar_constant * 
                           torch.exp(-optical_depth * air_mass) * 
                           cos_theta * cloud_transmission * atm)

        # Simplified diffuse irradiance calculation
        diffuse_irradiance = 0.3 * direct_irradiance * (1.0 - cos_zenith)

        # Total theoretical irradiance
        theoretical_irradiance = direct_irradiance + diffuse_irradiance

        # Simple efficiency clamping (15%-25%)
        y_pred = torch.clamp(y_pred, min=0.15, max=0.25)

        # Calculate residuals
        spatial_residual = torch.mean(torch.abs(torch.gradient(y_pred, dim=0)[0]))
        temporal_residual = torch.mean(torch.abs(torch.gradient(y_pred, dim=1)[0]))

        # Physics-based residual with simplified relative error
        physics_residual = torch.abs(y_pred - theoretical_irradiance) / (theoretical_irradiance + 1e-6)

        # Dynamic weighting with reduced physics weight
        confidence = torch.exp(-physics_residual.mean())
        spatial_weight = 0.3 * confidence
        temporal_weight = 0.3 * confidence
        physics_weight = 0.2  # Reduced fixed weight for physics residual

        # Total loss with simplified weights
        total_loss = (
            spatial_weight * spatial_residual +
            temporal_weight * temporal_residual +
            physics_weight * physics_residual.mean()
        )

        return total_loss


class PINNTrainer:

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        y_pred = self.model(x_data)
        
        # Compute losses
        data_loss = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        
        # Total loss with balanced weights
        total_loss = 0.5 * data_loss + 0.5 * physics_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), data_loss.item(), physics_loss.item()