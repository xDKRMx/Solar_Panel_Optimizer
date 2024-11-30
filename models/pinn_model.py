import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, input_dim=8):  # Updated for cloud_cover and wavelength
        super(SolarPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.solar_constant = 1367.0  # W/m²
        self.ref_wavelength = 0.5  # μm, reference wavelength for Ångström formula
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        self.cloud_alpha = 0.85  # Empirically derived cloud transmission parameter
        self.ref_temp = 25.0  # Reference temperature (°C)
        self.temp_coeff = 0.004  # Temperature coefficient (/°C)
        
    def forward(self, x):
        # Add tanh activation to constrain output to [-1, 1]
        return torch.tanh(self.net(x))
    
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
        
        # Calculate cos(zenith) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) +
                     torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle))
        
        # Calculate cos(θ) for tilted surface
        cos_theta = (torch.sin(lat_rad) * torch.sin(declination) * torch.cos(slope_rad) -
                    torch.cos(lat_rad) * torch.sin(declination) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.cos(slope_rad) +
                    torch.sin(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(declination) * torch.sin(hour_angle) * torch.sin(slope_rad) * torch.sin(aspect_rad))
        
        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)  # Avoid division by zero
    
    def calculate_optical_depth(self, wavelength, air_mass):
        """Calculate multi-wavelength optical depth with air mass dependency"""
        base_depth = self.beta * (wavelength / self.ref_wavelength) ** (-self.alpha)
        return base_depth * torch.exp(-base_depth * air_mass)
    
    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))
        
        # Compute gradients for physics residual
        y_grad = torch.autograd.grad(
            y_pred.sum(), x, 
            create_graph=True, retain_graph=True
        )[0]
        
        # Calculate theoretical irradiance based on Beer-Lambert law
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)
        
        # Air mass ratio calculation using Kasten and Young's formula
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))
        
        # Calculate optical depth and cloud cover effect
        optical_depth = self.calculate_optical_depth(wavelength, air_mass)
        cloud_transmission = 1.0 - self.cloud_alpha * (cloud_cover ** 3)

        # Calculate second-order derivatives for energy flux conservation
        y_grad2 = torch.autograd.grad(
            y_grad.sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        
        # Energy flux conservation: ∇⋅(q) + S = 0
        flux_divergence = y_grad2[:, 0] + y_grad2[:, 1]  # ∂²I/∂x² + ∂²I/∂y²
        source_term = y_grad[:, 2]  # ∂I/∂t (temporal variation as source)
        conservation_residual = flux_divergence + source_term
        
        # Beer-Lambert law implementation
        theoretical_irradiance = (self.solar_constant * 
                                torch.exp(-optical_depth * air_mass) * 
                                cos_theta * 
                                cloud_transmission)
        
        # Physics residuals
        spatial_residual = y_grad[:, 0]**2 + y_grad[:, 1]**2  # Spatial variation
        temporal_residual = y_grad[:, 2]  # Time variation
        physics_residual = y_pred - theoretical_irradiance  # Physics-based prediction
        # Add physical bounds constraint
        max_irradiance_residual = torch.relu(y_pred - self.solar_constant * cos_theta)
        boundary_residual = torch.relu(-y_pred)  # Non-negative constraint
        
        # Dynamic weighting based on training progress
        spatial_weight = 0.2 * torch.exp(-conservation_residual.abs().mean())
        temporal_weight = 0.2 * torch.exp(-temporal_residual.abs().mean())
        physics_weight = 0.4 * (1 - torch.exp(-physics_residual.abs().mean()))  # Increased from 0.3
        boundary_weight = 0.3  # Increased from 0.15
        conservation_weight = 0.15

        # Combine residuals with dynamic weights
        total_residual = (spatial_weight * spatial_residual + 
                         temporal_weight * temporal_residual + 
                         physics_weight * physics_residual**2 +
                         boundary_weight * (boundary_residual + max_irradiance_residual) +
                         conservation_weight * conservation_residual**2)
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return torch.mean(total_residual)

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        
        # Loss weights
        self.w_data = 0.4      # Weight for data fitting
        self.w_physics = 0.4   # Weight for physics constraints
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
        time_end = self.model(torch.cat([
            x_data[:, :2],
            (time + 24).unsqueeze(1),
            x_data[:, 3:]
        ], dim=1))
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
        total_loss = (self.w_data * mse + 
                     self.w_physics * physics_loss +
                     self.w_boundary * boundary_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mse.item(), physics_loss.item(), boundary_loss.item()
