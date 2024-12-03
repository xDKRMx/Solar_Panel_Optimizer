import numpy as np
import torch
import torch.nn as nn

class SolarPINN(nn.Module):
    def __init__(self, input_dim=8): 
        super(SolarPINN, self).__init__()
        # Simplified network architecture with 3 layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Essential physics coefficients
        self.solar_constant = 1367.0  # Solar constant (W/m²)
        self.beta = 0.125      # Atmospheric turbidity coefficient
        self.ref_temp = 25.0   # Reference temperature (°C)

    def forward(self, x):
        raw_output = self.net(x)
        return torch.sigmoid(raw_output)

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
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (x[:, i] for i in range(8))

        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)

        # Simple day/night validation
        nighttime_condition = (cos_zenith <= 0.001)
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred) * 100.0,
            torch.zeros_like(y_pred)
        )

        # Simple atmospheric transmission
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))
        
        # Direct solar irradiance only
        transmission = torch.exp(-self.beta * air_mass)
        theoretical_irradiance = self.solar_constant * transmission * cos_theta
        
        # Physics residual based on direct irradiance only
        physics_residual = torch.where(
            theoretical_irradiance < 1.0,
            torch.abs(y_pred) * 50.0,
            (y_pred - theoretical_irradiance)**2
        )

        # Efficiency constraints (15-25% range)
        efficiency_penalty = (
            torch.relu(0.15 - y_pred) + 
            torch.relu(y_pred - 0.25)
        ) * 100.0

        total_residual = (
            5.0 * physics_residual +
            10.0 * nighttime_penalty +
            efficiency_penalty
        )

        return torch.mean(total_residual)

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999)
        )
        self.mse_loss = nn.MSELoss()
        
        # Simplified loss weights
        self.w_data = 0.3
        self.w_physics = 0.7

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        y_pred = self.model(x_data)
        
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        total_loss = self.w_data * mse + self.w_physics * physics_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(), 0.0