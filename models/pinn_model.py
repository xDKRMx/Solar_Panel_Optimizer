import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):

    def __init__(self, input_dim=8, reg_scale=0.01, dropout_rate=0.2):  # Simplified parameters
        super(SolarPINN, self).__init__()
        self.reg_scale = reg_scale  # Reduced L2 regularization
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),  # Keep only one LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1)
        )
        self.solar_constant = 1365.0  # W/m²

    def forward(self, x):
        raw_output = self.net(x)
        return torch.clamp(raw_output, min=0.0, max=1.0)  # Simple output clipping

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
            cos_zenith, min=0.0001)

    def physics_loss(self, x, y_pred):
        """Simplified physics loss calculation"""
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))

        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(
            lat, lon, time, slope, aspect)

        # Simple nighttime penalty
        nighttime_penalty = torch.where(
            cos_zenith <= 0.001,
            torch.abs(y_pred) * 100.0,
            torch.zeros_like(y_pred)
        )

        # Calculate air mass and optical depth
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))
        optical_depth = 0.1 * (wavelength / 0.5).pow(-1.3)  # Simplified optical depth

        # Basic cloud effect
        cloud_transmission = 1.0 - 0.75 * cloud_cover

        # Theoretical irradiance (simplified)
        theoretical_irradiance = self.solar_constant * torch.exp(-optical_depth * air_mass) * cos_theta * cloud_transmission * atm

        # Basic physics residual
        physics_residual = torch.where(
            theoretical_irradiance < 1.0,
            torch.abs(y_pred),
            (y_pred - theoretical_irradiance).pow(2)
        )

        # Add L2 regularization only
        l2_reg = sum(torch.norm(param, p=2) for param in self.parameters())
        
        # Simplified total residual
        total_residual = (
            1.0 * physics_residual +  # Reduced weight
            0.1 * nighttime_penalty +  # Reduced penalty
            self.reg_scale * l2_reg  # Reduced L2 regularization
        )

        return torch.mean(total_residual)


class PINNTrainer:

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model(x_data)

        # Compute losses
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        # Total loss (simplified)
        total_loss = 0.5 * mse + 0.5 * physics_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item()