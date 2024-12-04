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
        
        # Simplified physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter

    def forward(self, x):
        return torch.sigmoid(self.net(x))

    def solar_declination(self, time):
        """Calculate solar declination angle (δ) with normalized time"""
        t_star = time / 24.0  # Normalize time to [0, 1] for daily cycle
        day_number = (t_star * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω) with normalized time"""
        t_star = time / 24.0  # Normalize time to [0, 1]
        hour = (t_star * 24) % 24
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

        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)

        # Nighttime constraint (zero irradiance when cos_zenith <= 0)
        nighttime_mask = (cos_zenith <= 0)
        y_pred = torch.where(nighttime_mask, torch.zeros_like(y_pred), y_pred)

        # Simplified air mass calculation
        air_mass = 1.0 / (cos_zenith + 1e-6)  # Simplified from Kasten-Young formula

        # Non-dimensionalized and simplified calculations
        wavelength_star = wavelength / self.lambda_ref
        optical_depth = self.beta * (wavelength_star)**(-self.alpha)
        cloud_transmission = 1.0 - self.cloud_alpha * cloud_cover

        # Simplified irradiance calculation (non-dimensionalized)
        irradiance_star = (torch.exp(-optical_depth * air_mass) * 
                          cos_theta * cloud_transmission * atm)

        # Simple min/max clipping for efficiency
        y_pred_clipped = torch.clamp(y_pred, min=0.15, max=0.25)
        efficiency_penalty = torch.mean(torch.abs(y_pred - y_pred_clipped))

        # Physics residual (simplified)
        physics_residual = torch.abs(y_pred - irradiance_star)

        # Total loss with simplified weights
        total_loss = (
            0.7 * physics_residual.mean() +  # Main physics component
            0.3 * efficiency_penalty  # Efficiency constraints
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