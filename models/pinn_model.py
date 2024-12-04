import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):

    def __init__(self, input_dim=10):  # Updated for altitude and temperature
        super(SolarPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Solar and atmospheric constants
        self.solar_constant = 1367.0  # W/m²
        self.ref_wavelength = 0.5  # μm, reference wavelength for Ångström formula
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        
        # Enhanced cloud physics parameters
        self.cloud_alpha = 0.85  # Cloud transmission parameter
        self.cloud_scatter = 0.2  # Cloud scattering coefficient
        self.cloud_absorption = 0.1  # Cloud absorption coefficient
        
        # Temperature coefficients
        self.ref_temp = 25.0  # Reference temperature (°C)
        self.temp_coeff = -0.004  # Temperature coefficient (/°C)
        self.temp_threshold = 45.0  # Maximum operating temperature
        
        # Surface albedo parameters
        self.base_albedo = 0.2  # Default ground albedo
        self.snow_albedo = 0.8  # Snow surface albedo
        self.water_albedo = 0.06  # Water surface albedo

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
        # Extract parameters from input including new ones
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength, altitude, temperature = (
            x[:, i] for i in range(10))

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

        # Air mass ratio calculation using improved Kasten and Young's formula with altitude
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        base_air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))
        # Altitude correction for air mass
        pressure_ratio = torch.exp(-altitude / 7.4)  # 7.4 km scale height
        air_mass = base_air_mass * pressure_ratio

        # Calculate optical depth with enhanced cloud and altitude physics
        optical_depth = self.calculate_optical_depth(wavelength,
                                                   air_mass,
                                                   altitude=altitude,
                                                   cloud_cover=cloud_cover)

        # Enhanced cloud transmission model with altitude and temperature dependency
        base_transmission = 1.0 - self.cloud_alpha * (cloud_cover**2)
        # Temperature effect on cloud formation
        temp_factor = torch.clamp((temperature - 20) / 30, 0, 1)  # Normalized temperature effect
        diffuse_factor = 0.3 * cloud_cover * (1.0 - cos_zenith) * (1 + 0.2 * temp_factor)
        cloud_transmission = base_transmission + diffuse_factor

        # Calculate second-order derivatives for energy flux conservation
        y_grad2 = torch.autograd.grad(y_grad.sum(),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)[0]

        # Enhanced energy flux conservation with temperature gradients
        flux_divergence = y_grad2[:, 0] + y_grad2[:, 1] + 0.1 * y_grad2[:, 9]  # Include temperature
        source_term = y_grad[:, 2]
        conservation_residual = flux_divergence + source_term

        # Enhanced theoretical irradiance components with temperature effects
        # Temperature efficiency factor
        temp_efficiency = 1.0 - self.temp_coeff * (temperature - self.ref_temp)
        temp_efficiency = torch.clamp(temp_efficiency, 0.8, 1.0)  # Limit temperature losses

        direct_irradiance = (self.solar_constant * 
                           torch.exp(-optical_depth * air_mass) * 
                           cos_theta * cloud_transmission * temp_efficiency)
        
        # Enhanced diffuse radiation model with altitude effects
        altitude_factor = torch.exp(-altitude / 15.0)  # Less diffuse at higher altitudes
        diffuse_factor = (0.3 + 0.7 * cloud_cover) * altitude_factor
        diffuse_irradiance = (self.solar_constant * diffuse_factor * 
                            (1.0 - cloud_transmission) * cos_zenith *
                            torch.exp(-optical_depth * air_mass * 0.5) * temp_efficiency)
        
        # Enhanced ground reflection model with surface type consideration
        snow_condition = temperature < 0
        water_condition = altitude < 0.01  # Near sea level
        ground_albedo = torch.where(
            snow_condition,
            self.snow_albedo,
            torch.where(
                water_condition,
                self.water_albedo,
                self.base_albedo + 0.1 * cloud_cover
            )
        )
        reflected_irradiance = (ground_albedo * (direct_irradiance + diffuse_irradiance) * 
                              (1.0 - cos_theta) * 0.5)

        # Total theoretical irradiance with temperature and altitude effects
        cos_zenith = torch.clamp(cos_zenith, min=0.001, max=1.0)
        theoretical_irradiance = torch.where(
            cos_zenith > 0,
            (direct_irradiance + diffuse_irradiance + reflected_irradiance) * temp_efficiency,
            torch.zeros_like(direct_irradiance))

        # Enhanced physics residuals with temperature consideration
        spatial_residual = y_grad[:, 0]**2 + y_grad[:, 1]**2 + 0.1 * y_grad[:, 9]**2
        temporal_residual = y_grad[:, 2]
        temp_residual = torch.abs(y_grad[:, 9])  # Temperature gradient penalty

        # Strengthen physics-based matching with temperature effects
        physics_residual = torch.where(
            theoretical_irradiance < 1.0,
            torch.abs(y_pred) * 500.0,
            (y_pred - theoretical_irradiance)**2
        )

        # Dynamic weighting with temperature consideration
        spatial_weight = 0.2 * torch.exp(-conservation_residual.abs().mean())
        temporal_weight = 0.2 * torch.exp(-temporal_residual.abs().mean())
        temp_weight = 0.1 * torch.exp(-temp_residual.mean())
        boundary_weight = 0.15
        conservation_weight = 0.15

        # Enhanced efficiency bounds considering temperature
        base_efficiency_min = 0.15
        base_efficiency_max = 0.25
        temp_adjustment = torch.clamp(0.02 * (temperature - self.ref_temp) / 20.0, -0.02, 0.02)
        efficiency_min = base_efficiency_min - temp_adjustment
        efficiency_max = base_efficiency_max - temp_adjustment
        
        # Refined efficiency penalties
        efficiency_lower = torch.exp(-100 * (y_pred - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred - efficiency_max))
        efficiency_penalty = (efficiency_lower + efficiency_upper) * 2000.0
        
        # Additional barrier terms with temperature consideration
        additional_lower_barrier = torch.exp(-200 * (y_pred - efficiency_min))
        additional_upper_barrier = torch.exp(200 * (y_pred - efficiency_max))
        efficiency_penalty += (additional_lower_barrier + additional_upper_barrier) * 5000.0

        # Enhanced clipping penalty with temperature adjustment
        clipping_penalty = torch.mean(torch.abs(
            y_pred - torch.clamp(y_pred, min=efficiency_min, max=efficiency_max)
        )) * 100.0

        # Total residual with enhanced weighting
        total_residual = (
            spatial_weight * spatial_residual +
            temporal_weight * temporal_residual +
            temp_weight * temp_residual +
            5.0 * physics_residual +
            boundary_weight * (torch.relu(-y_pred) + torch.relu(y_pred - self.solar_constant)) +
            conservation_weight * conservation_residual**2 +
            10.0 * nighttime_penalty +
            efficiency_penalty +
            clipping_penalty
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