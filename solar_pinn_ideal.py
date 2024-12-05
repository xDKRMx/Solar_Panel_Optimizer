import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics constraints for ideal conditions."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights with small random values for better convergence
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.tanh(self.linear(x))

class SolarPINNIdeal(nn.Module):
    """Simplified PINN for solar irradiance under ideal conditions."""
    def __init__(self, input_dim=4):  # [latitude, time, day_of_year, slope]
        super().__init__()
        self.solar_constant = 1367.0  # W/m²
        
        # Simplified network architecture
        self.network = nn.Sequential(
            PhysicsInformedLayer(input_dim, 64),
            PhysicsInformedLayer(64, 32),
            PhysicsInformedLayer(32, 16),
            nn.Linear(16, 1)
        )
        
    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle."""
        return 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)

    def calculate_hour_angle(self, time):
        """Calculate hour angle from local time."""
        return 15.0 * (time - 12.0)  # 15 degrees per hour

    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate solar zenith angle."""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        hour_rad = torch.deg2rad(hour_angle)
        
        cos_zenith = (torch.sin(lat_rad) * torch.sin(decl_rad) + 
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_rad))
        return torch.arccos(torch.clamp(cos_zenith, -1.0, 1.0))

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten & Young formula."""
        zenith_deg = torch.rad2deg(zenith_angle)
        denominator = (torch.cos(zenith_angle) + 
                      0.50572 * (96.07995 - zenith_deg) ** (-1.6364))
        return torch.where(zenith_deg < 90, 1.0 / denominator, float('inf'))

    def calculate_surface_factor(self, zenith_angle, slope):
        """Calculate surface orientation factor for ideal conditions."""
        slope_rad = torch.deg2rad(slope)
        cos_incidence = torch.cos(zenith_angle) * torch.cos(slope_rad) + \
                       torch.sin(zenith_angle) * torch.sin(slope_rad)
        return torch.clamp(cos_incidence, 0.0, 1.0)

    def calculate_day_night(self, latitude, declination, time):
        """Calculate day/night mask."""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(decl_rad)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1.0, 1.0)
        
        sunrise_angle = torch.arccos(cos_hour_angle)
        sunrise_hour = 12.0 - (torch.rad2deg(sunrise_angle) / 15.0)
        sunset_hour = 12.0 + (torch.rad2deg(sunrise_angle) / 15.0)
        
        return (time >= sunrise_hour) & (time <= sunset_hour)

    def calculate_clear_sky_irradiance(self, zenith_angle, air_mass, surface_factor):
        """Calculate theoretical clear sky irradiance."""
        cos_zenith = torch.cos(zenith_angle)
        atmospheric_transmission = torch.exp(-0.1 * air_mass)  # Simple clear-sky model
        return self.solar_constant * cos_zenith * atmospheric_transmission * surface_factor

    def forward(self, x):
        """Forward pass with physics constraints."""
        # Extract input components
        latitude = x[:, 0]
        time = x[:, 1]
        day_of_year = x[:, 2]
        slope = x[:, 3]

        # Calculate physical parameters
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time)
        zenith_angle = self.calculate_zenith_angle(latitude, declination, hour_angle)
        air_mass = self.calculate_air_mass(zenith_angle)
        surface_factor = self.calculate_surface_factor(zenith_angle, slope)
        
        # Neural network prediction
        prediction = self.network(x)
        
        # Apply physical constraints
        clear_sky = self.calculate_clear_sky_irradiance(zenith_angle, air_mass, surface_factor)
        constrained_prediction = torch.sigmoid(prediction) * clear_sky.unsqueeze(-1)
        
        # Apply day/night mask
        day_mask = self.calculate_day_night(latitude, declination, time)
        final_prediction = torch.where(
            day_mask.unsqueeze(-1),
            constrained_prediction,
            torch.zeros_like(constrained_prediction)
        )
        
        return final_prediction

class SolarPINNTrainer:
    """Trainer class with validation metrics."""
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
    
    def calculate_metrics(self, y_pred, y_true):
        """Calculate validation metrics."""
        mse = F.mse_loss(y_pred, y_true)
        mae = F.l1_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        
        # Calculate R² score
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'rmse': rmse.item(),
            'r2': r2.item()
        }

    def train_step(self, x_batch, y_batch, validation=False):
        """Training step with validation option."""
        self.model.train(not validation)
        
        with torch.set_grad_enabled(not validation):
            # Forward pass
            y_pred = self.model(x_batch)
            
            # Calculate losses
            data_loss = F.mse_loss(y_pred, y_batch)
            
            if not validation:
                # Optimize
                self.optimizer.zero_grad()
                data_loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_pred, y_batch)
            
        return data_loss.item(), metrics
