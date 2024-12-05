import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):

    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        # Enhanced neural network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU(),
            nn.Linear(128, 64), nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        # Reference values for non-dimensionalization
        self.I0 = 1367.0  # W/m² (solar constant)
        self.lambda_ref = 0.5  # μm (reference wavelength)
        self.T_ref = 298.15  # K (25°C reference temperature)
        self.time_ref = 24.0  # hours
        self.length_ref = 1000.0  # meters
        
        # Enhanced physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter
        self.terrain_resolution = 100  # meters per grid point
        self.albedo = 0.2  # Ground reflectance
        
        # Initialize terrain model
        from .terrain_model import TerrainModel
        self.terrain_model = TerrainModel()

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
        """Calculate solar declination angle (δ) with enhanced seasonal effects"""
        t_star = time / self.time_ref
        day_number = (t_star * 365).clamp(0, 365)
        
        # Enhanced seasonal calculation with elliptical orbit correction
        gamma = 2 * np.pi * (day_number - 1) / 365
        delta = 0.006918 - 0.399912 * torch.cos(gamma) + 0.070257 * torch.sin(gamma) \
               - 0.006758 * torch.cos(2*gamma) + 0.000907 * torch.sin(2*gamma) \
               - 0.002697 * torch.cos(3*gamma) + 0.001480 * torch.sin(3*gamma)
        
        return delta

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω) with equation of time correction"""
        t_star = time / self.time_ref
        day_number = (t_star * 365).clamp(0, 365)
        
        # Equation of time correction
        b = 2 * np.pi * (day_number - 81) / 365
        eot = 9.87 * torch.sin(2*b) - 7.53 * torch.cos(b) - 1.5 * torch.sin(b)
        
        # Solar time calculation
        hour = (t_star * self.time_ref) % self.time_ref
        solar_time = hour + eot/60 + lon/15
        
        return torch.deg2rad(15 * (solar_time - 12))

    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ) and zenith angle with improved accuracy"""
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

        # Calculate solar position and angles
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)
        solar_zenith = torch.acos(cos_zenith)
        solar_azimuth = self.calculate_azimuth(lat, lon, time)

        # Calculate terrain shading
        terrain_shading = self.terrain_model.compute_terrain_shading(x, solar_zenith, solar_azimuth)
        
        # Calculate diffuse radiation
        diffuse_radiation = self.terrain_model.compute_diffuse_radiation(
            x, self.T_ref * torch.ones_like(lat), 101325 * torch.ones_like(lat)
        )

        # Nighttime constraint (zero irradiance when cos_zenith <= 0)
        nighttime_mask = (cos_zenith <= 0)
        y_pred = torch.where(nighttime_mask, torch.zeros_like(y_pred), y_pred)

        # Enhanced air mass calculation (Kasten-Young formula)
        zenith_rad = torch.acos(cos_zenith)
        air_mass = 1 / (cos_zenith + 0.50572 * (96.07995 - zenith_rad.rad2deg())**(-1.6364))
        
        # Advanced atmospheric modeling
        # Linke turbidity factor (seasonal variation)
        day_angle = 2 * np.pi * t_star
        linke_turbidity = 2 + 0.5 * torch.sin(day_angle - np.pi/2)
        
        # Rayleigh scattering
        rayleigh = torch.exp(-0.1184 * air_mass)
        
        # Aerosol extinction with wavelength dependency
        optical_depth = self.beta * (lambda_star)**(-self.alpha)
        aerosol = torch.exp(-optical_depth * air_mass)
        
        # Enhanced cloud model with multiple layer effects
        cloud_transmission = 1.0 - self.cloud_alpha * (cloud_cover ** 3)
        
        # Total atmospheric transmission
        atmospheric_transmission = rayleigh * aerosol * cloud_transmission * atm

        # Direct and diffuse components
        direct_irradiance = self.I0 * cos_theta * atmospheric_transmission * terrain_shading
        diffuse_irradiance = self.I0 * diffuse_radiation * (1 + self.albedo * cloud_cover)
        total_irradiance = direct_irradiance + diffuse_irradiance
        
        # Non-dimensionalize irradiance
        irradiance_star = total_irradiance / self.I0

        # Physical constraints
        efficiency_min, efficiency_max = 0.15, 0.25
        efficiency_lower = torch.exp(-100 * (y_pred - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred - efficiency_max))
        efficiency_penalty = efficiency_lower + efficiency_upper

        # Enhanced physics residual calculation
        physics_residual = torch.abs(y_pred - irradiance_star) + \
                          torch.abs(torch.gradient(y_pred, dim=0)[0]) + \
                          0.1 * torch.abs(torch.gradient(y_pred, dim=1)[0])

        # Terrain influence penalty
        terrain_penalty = torch.mean(torch.abs(
            torch.gradient(y_pred * terrain_shading, dim=0)[0]
        ))

        # Dynamic loss weights
        physics_weight = torch.exp(-physics_residual.mean())
        efficiency_weight = torch.exp(-efficiency_penalty)
        terrain_weight = torch.exp(-terrain_penalty)

        # Total loss with dynamic weights
        total_loss = (
            0.4 * physics_weight * physics_residual.mean() +
            0.4 * efficiency_weight * efficiency_penalty +
            0.2 * terrain_weight * terrain_penalty
        )

        return total_loss

    def calculate_azimuth(self, lat, lon, time):
        """Calculate solar azimuth angle"""
        declination = self.solar_declination(time)
        hour_angle = self.hour_angle(time, lon)
        lat_rad = torch.deg2rad(lat)
        
        cos_zenith = torch.sin(lat_rad) * torch.sin(declination) + \
                     torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle)
        sin_zenith = torch.sqrt(1 - cos_zenith**2)
        
        cos_azimuth = (torch.sin(declination) - torch.sin(lat_rad) * cos_zenith) / \
                      (torch.cos(lat_rad) * sin_zenith + 1e-6)
        sin_azimuth = cos_declination * torch.sin(hour_angle) / sin_zenith
        
        azimuth = torch.atan2(sin_azimuth, cos_azimuth)
        return azimuth

    def denormalize_predictions(self, y_pred_star):
        """Convert non-dimensional predictions back to physical units"""
        return y_pred_star * self.I0  # Scale back to W/m²


class PINNTrainer:

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
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
        
        # Adaptive loss weighting based on training progress
        loss_ratio = data_loss.item() / (physics_loss.item() + 1e-8)
        alpha = torch.sigmoid(torch.tensor(loss_ratio))
        
        # Total loss with adaptive weights
        total_loss = alpha * data_loss + (1 - alpha) * physics_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)

        return total_loss.item(), data_loss.item(), physics_loss.item()
