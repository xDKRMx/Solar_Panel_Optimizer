import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super(PINN, self).__init__()
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
        # Physical constants and reference values
        self.I0 = 1367.0  # W/m² (solar constant)
        self.lambda_ref = 0.5  # μm (reference wavelength)
        self.T_ref = 298.15  # K (25°C reference temperature)
        self.zenith_ref = torch.deg2rad(torch.tensor(23.45))  # Reference zenith angle
        self.tau_ref = 1.0  # Reference optical depth
        
        # Simplified physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter
        
        # Normalization ranges
        self.lat_scale = 90.0  # Latitude range [-90, 90] -> [-1, 1]
        self.lon_scale = 180.0  # Longitude range [-180, 180] -> [-1, 1]
        self.angle_scale = 360.0  # Angles [0, 360] -> [0, 1]

    def forward(self, x):
        return self.net(x)

    def normalize_inputs(self, lat, lon, time, slope, aspect, atm, cloud_cover, wavelength):
        """Normalize input parameters to appropriate ranges"""
        lat_norm = lat / self.lat_scale  # [-1, 1]
        lon_norm = lon / self.lon_scale  # [-1, 1]
        time_norm = time / 24.0  # [0, 1]
        slope_norm = slope / self.angle_scale  # [0, 1]
        aspect_norm = aspect / self.angle_scale  # [0, 1]
        atm_norm = atm  # Already normalized [0, 1]
        # cloud_cover and wavelength are already normalized
        return lat_norm, lon_norm, time_norm, slope_norm, aspect_norm, atm_norm, cloud_cover, wavelength

    def denormalize_predictions(self, y_pred_normalized):
        """Convert normalized predictions back to physical units"""
        return y_pred_normalized * self.I0  # Scale back to W/m²

    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle and zenith angle"""
        # Convert angles to radians
        lat_rad = torch.deg2rad(lat)
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        
        # Calculate solar angles
        hour_angle = 2 * np.pi * (time / 24.0 - 0.5)
        declination = 0.409 * torch.sin(2 * np.pi * (time / 24.0 - 0.25))
        
        # Calculate cosine of zenith angle
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) +
                     torch.cos(lat_rad) * torch.cos(declination) *
                     torch.cos(hour_angle))
        
        # Calculate cosine of incidence angle
        cos_theta = (torch.sin(declination) * torch.sin(lat_rad) *
                    torch.cos(slope_rad) -
                    torch.cos(lat_rad) * torch.sin(declination) *
                    torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(lat_rad) * torch.cos(declination) *
                    torch.cos(hour_angle) * torch.cos(slope_rad) +
                    torch.sin(lat_rad) * torch.cos(declination) *
                    torch.cos(hour_angle) * torch.sin(slope_rad) *
                    torch.cos(aspect_rad) +
                    torch.cos(declination) * torch.sin(hour_angle) *
                    torch.sin(slope_rad) * torch.sin(aspect_rad))

        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)

    def physics_loss(self, x, y_pred):
        """Calculate physics-informed loss with enhanced constraints"""
        # Extract and normalize parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))
        lat_norm, lon_norm, time_norm, slope_norm, aspect_norm, atm_norm, cloud_cover, wavelength = \
            self.normalize_inputs(lat, lon, time, slope, aspect, atm, cloud_cover, wavelength)

        # Calculate cos_theta and cos_zenith using normalized inputs
        cos_theta, cos_zenith = self.cos_incidence_angle(
            lat_norm * self.lat_scale,
            lon_norm * self.lon_scale,
            time_norm * 24.0,
            slope_norm * self.angle_scale,
            aspect_norm * self.angle_scale
        )

        # Nighttime penalty using soft constraint
        nighttime_penalty = torch.where(
            cos_zenith <= 0.001,
            1000.0 * torch.abs(y_pred),  # Strong penalty for non-zero predictions
            torch.zeros_like(y_pred)
        )

        # Non-dimensionalized air mass calculation (Kasten-Young formula)
        air_mass = 1.0 / (cos_zenith + 1e-6)
        air_mass_star = air_mass / torch.cos(self.zenith_ref)

        # Non-dimensionalized optical depth calculation
        wavelength_star = wavelength / self.lambda_ref
        optical_depth = self.beta * (wavelength_star)**(-self.alpha)
        optical_depth_star = optical_depth / self.tau_ref

        # Simplified cloud transmission (linear model)
        cloud_transmission = 1.0 - self.cloud_alpha * cloud_cover

        # Dynamic diffuse factor based on cloud cover
        diffuse_factor = 0.3 + 0.7 * cloud_cover  # Increased scattering under cloudy conditions

        # Non-dimensionalized irradiance calculation with enhanced diffuse component
        direct_irradiance = torch.exp(-optical_depth_star * air_mass_star) * cos_theta * cloud_transmission
        diffuse_irradiance = diffuse_factor * direct_irradiance * (1 - cos_zenith)
        irradiance_star = (direct_irradiance + diffuse_irradiance) * atm_norm

        # Efficiency constraints with exponential barriers
        efficiency_min, efficiency_max = 0.15, 0.25
        efficiency_lower = torch.exp(-100 * (y_pred/self.I0 - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred/self.I0 - efficiency_max))
        efficiency_penalty = efficiency_lower + efficiency_upper

        # Calculate residuals
        physics_residual = torch.abs(y_pred/self.I0 - irradiance_star)
        spatial_residual = torch.abs(torch.gradient(y_pred, dim=0)[0])
        temporal_residual = 0.1 * torch.abs(torch.gradient(y_pred, dim=1)[0])

        # Dynamic weights based on residual magnitudes
        physics_weight = torch.exp(-physics_residual.mean())
        spatial_weight = torch.exp(-spatial_residual.abs().mean())
        temporal_weight = torch.exp(-temporal_residual.abs().mean())

        # Total loss with dynamic weights
        total_loss = (
            physics_weight * physics_residual.mean() +
            spatial_weight * spatial_residual.mean() +
            temporal_weight * temporal_residual.mean() +
            0.1 * efficiency_penalty.mean() +
            nighttime_penalty.mean()
        )

        return total_loss


class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        # Forward pass with normalized inputs
        x_normalized = torch.stack(self.model.normalize_inputs(*[x_data[:, i] for i in range(8)]), dim=1)
        y_pred = self.model(x_normalized)
        y_pred = self.model.denormalize_predictions(y_pred)
        
        # Compute losses
        data_loss = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        
        # Total loss with balanced weights
        total_loss = 0.5 * data_loss + 0.5 * physics_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), data_loss.item(), physics_loss.item()
