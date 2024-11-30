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
        self.cloud_alpha = 0.85  # Empirically derived cloud transmission parameter
        self.ref_temp = 25.0  # Reference temperature (°C)
        self.temp_coeff = 0.004  # Temperature coefficient (/°C)

    def forward(self, x):
        raw_output = self.net(x)
        # Let data_processor handle efficiency clipping
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

        # Calculate cos(zenith) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) +
                      torch.cos(lat_rad) * torch.cos(declination) *
                      torch.cos(hour_angle))

        # Calculate cos(θ) for tilted surface
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

    def calculate_optical_depth(self,
                                wavelength,
                                air_mass,
                                altitude=0,
                                cloud_cover=0):
        """Calculate advanced multi-wavelength optical depth with cloud and altitude effects"""
        # Base optical depth using Ångström turbidity formula
        base_depth = self.beta * (wavelength /
                                  self.ref_wavelength)**(-self.alpha)

        # Altitude correction (exponential decrease with height)
        altitude_factor = torch.exp(-altitude / 7.4)  # 7.4 km scale height

        # Enhanced cloud model with altitude dependency
        cloud_factor = 1.0 + (cloud_cover**2) * (1.0 - 0.5 * altitude_factor)

        # Air mass dependency with cloud-modified saturation
        air_mass_factor = torch.exp(-base_depth * air_mass * altitude_factor *
                                    cloud_factor)

        # Additional wavelength-dependent cloud scattering
        cloud_scatter = 0.2 * cloud_cover * (wavelength /
                                             self.ref_wavelength)**(-0.75)

        return base_depth * air_mass_factor + cloud_scatter

    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))

        # Calculate cos_theta and cos_zenith
        cos_theta, cos_zenith = self.cos_incidence_angle(
            lat, lon, time, slope, aspect)

        # Enhanced nighttime penalty with smooth transition
        nighttime_threshold = 0.001
        twilight_threshold = 0.1
        nighttime_condition = (cos_zenith <= nighttime_threshold)
        twilight_condition = (cos_zenith > nighttime_threshold) & (cos_zenith <= twilight_threshold)
        
        # Stronger penalty for nighttime predictions
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred) * 1000.0,  # Increased penalty for nighttime
            torch.where(
                twilight_condition,
                torch.abs(y_pred - self.solar_constant * cos_zenith) * 100.0,  # Smooth transition
                torch.zeros_like(y_pred)
            ))

        # Compute gradients for physics residual
        y_grad = torch.autograd.grad(y_pred.sum(),
                                   x,
                                   create_graph=True,
                                   retain_graph=True)[0]

        # Air mass ratio calculation using Kasten and Young's formula
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))

        # Calculate optical depth with enhanced cloud physics
        optical_depth = self.calculate_optical_depth(wavelength,
                                                   air_mass,
                                                   altitude=0,
                                                   cloud_cover=cloud_cover)

        # Enhanced cloud transmission model with altitude dependency
        base_transmission = 1.0 - self.cloud_alpha * (cloud_cover**2)
        diffuse_factor = 0.3 * cloud_cover * (1.0 - cos_zenith)
        cloud_transmission = base_transmission + diffuse_factor

        # Calculate second-order derivatives for energy flux conservation
        y_grad2 = torch.autograd.grad(y_grad.sum(),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)[0]

        # Energy flux conservation
        flux_divergence = y_grad2[:, 0] + y_grad2[:, 1]
        source_term = y_grad[:, 2]
        conservation_residual = flux_divergence + source_term

        # Enhanced theoretical irradiance components for full spectrum
        direct_irradiance = (self.solar_constant * 
                           torch.exp(-optical_depth * air_mass) * 
                           cos_theta * cloud_transmission)
        
        # Enhanced diffuse radiation model
        diffuse_factor = 0.3 + 0.7 * cloud_cover  # Increased diffuse component with clouds
        diffuse_irradiance = (self.solar_constant * diffuse_factor * 
                            (1.0 - cloud_transmission) * cos_zenith *
                            torch.exp(-optical_depth * air_mass * 0.5))  # Less attenuation for diffuse
        
        # Enhanced ground reflection model
        ground_albedo = 0.2 + 0.1 * cloud_cover  # Higher albedo under cloudy conditions
        reflected_irradiance = (ground_albedo * (direct_irradiance + diffuse_irradiance) * 
                              (1.0 - cos_theta) * 0.5)

        # Ensure proper cos_zenith clipping and set nighttime irradiance
        cos_zenith = torch.clamp(cos_zenith, min=0.001, max=1.0)
        theoretical_irradiance = torch.where(
            cos_zenith > 0,
            direct_irradiance + diffuse_irradiance + reflected_irradiance,
            torch.zeros_like(direct_irradiance))

        # Strengthen nighttime constraint with increased penalty
        nighttime_penalty = torch.where(
            cos_zenith <= 0.001,
            torch.abs(y_pred) * 1000.0,  # Increased penalty for non-zero nighttime predictions
            torch.zeros_like(y_pred)
        )

        # Physics residuals
        spatial_residual = y_grad[:, 0]**2 + y_grad[:, 1]**2
        temporal_residual = y_grad[:, 2]

        # Strengthen physics-based matching
        physics_residual = torch.where(
            theoretical_irradiance < 1.0,  # Near-zero physics-based irradiance
            torch.abs(y_pred) * 500.0,    # Strong penalty for non-zero predictions
            (y_pred - theoretical_irradiance)**2
        )

        # Dynamic weighting
        spatial_weight = 0.2 * torch.exp(-conservation_residual.abs().mean())
        temporal_weight = 0.2 * torch.exp(-temporal_residual.abs().mean())
        boundary_weight = 0.15
        conservation_weight = 0.15

        # Update efficiency bounds for expanded range
        efficiency_min = 0.15  # 15%
        efficiency_max = 0.25  # 25%
        
        # Exponential barrier functions for smoother gradients
        efficiency_lower = torch.exp(-100 * (y_pred - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred - efficiency_max))
        
        # Increased penalty weight for stricter enforcement (2000.0)
        efficiency_penalty = (efficiency_lower + efficiency_upper) * 2000.0
        
        # Additional exponential barrier terms with higher penalties (5000.0)
        additional_lower_barrier = torch.exp(-200 * (y_pred - efficiency_min))
        additional_upper_barrier = torch.exp(200 * (y_pred - efficiency_max))
        efficiency_penalty += (additional_lower_barrier + additional_upper_barrier) * 5000.0

        # Update clipping penalty
        clipping_penalty = torch.mean(torch.abs(
            y_pred - torch.clamp(y_pred, min=0.15, max=0.25)
        )) * 100.0

        # Update total_residual calculation
        total_residual = (
            spatial_weight * spatial_residual +
            temporal_weight * temporal_residual +
            5.0 * physics_residual +
            boundary_weight * (torch.relu(-y_pred) + torch.relu(y_pred - self.solar_constant)) +
            conservation_weight * conservation_residual**2 +
            10.0 * nighttime_penalty +
            efficiency_penalty +  # Increased penalty weight
            clipping_penalty  # Add hard clipping penalty
        )

        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return torch.mean(total_residual)


class PINNTrainer:

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

        # Loss weights
        self.w_data = 0.4  # Weight for data fitting
        self.w_physics = 0.4  # Weight for physics constraints
        self.w_boundary = 0.2  # Weight for boundary conditions

    def boundary_loss(self, x_data, y_pred):
        """Compute boundary condition losses"""
        # Night-time constraint (no negative irradiance)
        night_loss = torch.mean(torch.relu(-y_pred))

        # Maximum irradiance constraint
        max_loss = torch.mean(torch.relu(y_pred - self.model.solar_constant))

        # Time periodicity constraint
        time = x_data[:, 2]
        time_start = self.model(x_data.clone().detach().requires_grad_(True))
        time_end = self.model(
            torch.cat([x_data[:, :2], (time + 24).unsqueeze(1), x_data[:, 3:]],
                      dim=1))
        periodicity_loss = torch.mean((time_start - time_end)**2)

        return night_loss + max_loss + periodicity_loss

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model(x_data)

        # Compute losses
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        boundary_loss = self.boundary_loss(x_data, y_pred)

        # Total loss with weights
        total_loss = (self.w_data * mse + self.w_physics * physics_loss +
                      self.w_boundary * boundary_loss)

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(
        ), boundary_loss.item()
