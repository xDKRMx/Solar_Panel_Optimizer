import torch
import torch.nn as nn
import numpy as np

class SolarPINN(nn.Module):
    def __init__(self, input_dim=6):
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
        
    def forward(self, x):
        return self.net(x)
    
    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))
    
    def hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        hour = (time % 24).clamp(0, 24)
        return torch.deg2rad(15 * (hour - 12) + lon)
    
    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ)"""
        lat_rad = torch.deg2rad(lat)
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        
        declination = self.solar_declination(time)
        hour_angle = self.hour_angle(time, lon)
        
        # cos(θ) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
        cos_theta = (torch.sin(lat_rad) * torch.sin(declination) * torch.cos(slope_rad) -
                    torch.cos(lat_rad) * torch.sin(declination) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.cos(slope_rad) +
                    torch.sin(lat_rad) * torch.cos(declination) * torch.cos(hour_angle) * torch.sin(slope_rad) * torch.cos(aspect_rad) +
                    torch.cos(declination) * torch.sin(hour_angle) * torch.sin(slope_rad) * torch.sin(aspect_rad))
        
        return torch.clamp(cos_theta, min=0.0)
    
    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        
        # Compute gradients for physics residual
        y_grad = torch.autograd.grad(
            y_pred.sum(), x, 
            create_graph=True, retain_graph=True
        )[0]
        
        # Calculate theoretical irradiance based on physics
        cos_theta = self.cos_incidence_angle(lat, lon, time, slope, aspect)
        theoretical_irradiance = self.solar_constant * cos_theta * atm
        
        # Physics residuals
        spatial_residual = y_grad[:, 0]**2 + y_grad[:, 1]**2  # Spatial variation
        temporal_residual = y_grad[:, 2]  # Time variation
        physics_residual = y_pred - theoretical_irradiance  # Physics-based prediction
        boundary_residual = torch.relu(-y_pred)  # Non-negative constraint
        
        # Combine residuals with appropriate weights
        total_residual = (0.3 * spatial_residual + 
                         0.3 * temporal_residual + 
                         0.3 * physics_residual**2 +
                         0.1 * boundary_residual)
        
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
