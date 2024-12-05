import numpy as np
import torch
import torch.nn as nn


class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics constraints"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.physics_weights = nn.Parameter(
            torch.randn(out_features)
        )

    def forward(self, x):
        # Linear transformation
        out = self.linear(x)
        # Physics-based transformation
        out = out * torch.sigmoid(self.physics_weights)
        return out
import numpy as np
import torch
import torch.nn as nn


class SolarPINN(nn.Module):
    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        # Physics-informed architecture
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.Tanh(),  # Better for physics than LeakyReLU
            PhysicsInformedLayer(128, 256),
            nn.Tanh(),
            PhysicsInformedLayer(256, 1)
        )
        
        # Physical constants
        self.solar_constant = 1367.0  # W/m²
        self.stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
        
        # Physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter
        self.albedo = 0.2  # Ground reflectance
        
        # Initialize terrain model
        from .terrain_model import TerrainModel
        self.terrain_model = TerrainModel()

    def forward(self, x):
        # Unpack input parameters
        lat, lon, time, slope, aspect, atm, cloud, wavelength = x.split(1, dim=1)
        
        # Physics-informed forward pass
        return self.physics_net(x)
    
    def radiative_transfer_equation(self, x, I):
        """Core PDE: Radiative Transfer Equation"""
        # Automatic differentiation for spatial gradients
        I_gradients = torch.autograd.grad(
            I, x, 
            grad_outputs=torch.ones_like(I),
            create_graph=True
        )[0]
        
        # RTE components
        extinction = self.calculate_extinction(x)
        emission = self.calculate_emission(x)
        scattering = self.calculate_scattering(x, I)
        
        # RTE: dI/ds = -extinction*I + emission + scattering
        rte_residual = I_gradients + extinction*I - emission - scattering
        return rte_residual
    
    def calculate_extinction(self, x):
        """Calculate extinction coefficient"""
        _, _, _, _, _, atm, cloud, wavelength = x.split(1, dim=1)
        return self.beta * (wavelength)**(-self.alpha) * atm * (1 - self.cloud_alpha * cloud**3)

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

    def boundary_conditions(self, x):
        """Physical boundary conditions"""
        # Unpack parameters
        lat, lon, time, slope, aspect, atm, cloud, wavelength = x.split(1, dim=1)
        
        # Top of atmosphere condition (solar constant)
        cos_zenith = self.calculate_zenith_angle(lat, lon, time)
        toa_condition = self.solar_constant * cos_zenith
        
        # Surface boundary condition (albedo and emission)
        surface_temp = 288.15  # Standard surface temperature (K)
        surface_emission = self.stefan_boltzmann * (surface_temp**4)
        surface_reflection = self.albedo * toa_condition
        surface_condition = surface_reflection + surface_emission
        
        return toa_condition, surface_condition
    
    def check_energy_conservation(self, x, y_pred):
        """Verify energy conservation"""
        toa_condition, surface_condition = self.boundary_conditions(x)
        energy_balance = y_pred - (toa_condition - surface_condition)
        return energy_balance
    
    def calculate_emission(self, x):
        """Calculate thermal emission"""
        surface_temp = 288.15  # Standard surface temperature (K)
        return self.stefan_boltzmann * (surface_temp**4)
    
    def calculate_scattering(self, x, I):
        """Calculate scattering term"""
        _, _, _, _, _, atm, cloud, _ = x.split(1, dim=1)
        return self.albedo * atm * (1 - cloud) * I
    
    def physics_loss(self, x, y_pred):
        """Complete physics-informed loss"""
        # PDE residual
        rte_residual = self.radiative_transfer_equation(x, y_pred)
        
        # Boundary conditions
        toa_condition, surface_condition = self.boundary_conditions(x)
        
        # Conservation laws
        energy_conservation = self.check_energy_conservation(x, y_pred)
        
        # Combine all physics losses with weights
        physics_loss = (
            torch.mean(rte_residual**2) + 
            torch.mean((y_pred - toa_condition)**2) + 
            torch.mean((y_pred - surface_condition)**2) + 
            torch.mean(energy_conservation**2)
        )
        
        return physics_loss

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
