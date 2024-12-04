import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):

    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(),
                               nn.Linear(64, 128), nn.Tanh(),
                               nn.Linear(128, 64), nn.Tanh(),
                               nn.Linear(64, 1))
        # Reference values for non-dimensionalization
        self.I0 = 1367.0  # W/m² (solar constant)
        self.lambda_ref = 0.5  # μm (reference wavelength)
        self.T_ref = 298.15  # K (25°C reference temperature)
        self.time_ref = 24.0  # hours
        self.length_ref = 1000.0  # meters
        
        # Simplified physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter

    def forward(self, x):
        # Extract and non-dimensionalize inputs
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))
        
        # Non-dimensionalize inputs
        t_star = time / self.time_ref
        lambda_star = wavelength / self.lambda_ref
        
        # Reconstruct non-dimensionalized input tensor
        x_star = torch.stack([
            lat, lon, t_star, slope, aspect, atm, cloud_cover, lambda_star
        ], dim=1)
        
        return torch.sigmoid(self.net(x_star))

    def solar_declination(self, time):
        """Calculate solar declination angle (δ) with normalized time"""
        t_star = time / self.time_ref  # Normalize time to [0, 1] for daily cycle
        day_number = (t_star * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω) with normalized time"""
        t_star = time / self.time_ref  # Normalize time to [0, 1]
        hour = (t_star * self.time_ref) % self.time_ref
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

        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)

    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))

        # Non-dimensionalize inputs
        t_star = time / self.time_ref
        lambda_star = wavelength / self.lambda_ref

        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)

        # Nighttime constraint (zero irradiance when cos_zenith <= 0)
        nighttime_mask = (cos_zenith <= 0)
        y_pred = torch.where(nighttime_mask, torch.zeros_like(y_pred), y_pred)

        # Simplified air mass calculation
        air_mass = 1.0 / (cos_zenith + 1e-6)  # Simplified from Kasten-Young formula

        # Non-dimensionalized and simplified calculations
        optical_depth = self.beta * (lambda_star)**(-self.alpha)
        
        # Updated cloud transmission with cubic dependency
        cloud_transmission = 1.0 - self.cloud_alpha * (cloud_cover ** 3)

        # Non-dimensionalized irradiance calculation (I/I₀)
        irradiance_star = (torch.exp(-optical_depth * air_mass) * 
                          cos_theta * cloud_transmission * atm)

        # Exponential barrier functions for efficiency constraints
        efficiency_min, efficiency_max = 0.15, 0.25
        efficiency_lower = torch.exp(-100 * (y_pred - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred - efficiency_max))
        efficiency_penalty = efficiency_lower + efficiency_upper

        # Enhanced physics residual calculation with non-dimensional quantities
        physics_residual = torch.abs(y_pred - irradiance_star) + \
                          torch.abs(torch.gradient(y_pred, dim=0)[0]) + \
                          0.1 * torch.abs(torch.gradient(y_pred, dim=1)[0])

        # Dynamic loss weights
        physics_weight = torch.exp(-physics_residual.mean())
        efficiency_weight = torch.exp(-efficiency_penalty)

        # Total loss with dynamic weights
        total_loss = (
            physics_weight * physics_residual.mean() +
            efficiency_weight * efficiency_penalty
        )

        return total_loss

    def denormalize_predictions(self, y_pred_star):
        """Convert non-dimensional predictions back to physical units"""
        return y_pred_star * self.I0  # Scale back to W/m²


class PINNTrainer:

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        # Forward pass (produces non-dimensional predictions)
        y_pred_star = self.model(x_data)
        
        # Non-dimensionalize target data for comparison
        y_data_star = y_data / self.model.I0
        
        # Compute losses using non-dimensional quantities
        data_loss = self.mse_loss(y_pred_star, y_data_star)
        physics_loss = self.model.physics_loss(x_data, y_pred_star)
        
        # Total loss with balanced weights
        total_loss = 0.5 * data_loss + 0.5 * physics_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), data_loss.item(), physics_loss.item()
