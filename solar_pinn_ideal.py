import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics-informed constraints for solar irradiance prediction."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights with consideration for solar physics
        self.physics_weights = nn.Parameter(torch.ones(out_features) * 0.5)

    def forward(self, x):
        out = self.linear(x)
        # Ensure positive output through physics-based transformation
        out = out * torch.sigmoid(self.physics_weights)
        return out

class SolarPINN(nn.Module):
    def __init__(self, input_dim=5):  # latitude, longitude, time, slope, aspect
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
    
    def setup_physical_constants(self):
        """Set up essential physical constants for ideal conditions."""
        self.solar_constant = 1367.0  # Solar constant at top of atmosphere (W/m²)
        self.atmospheric_extinction = 0.1  # Idealized clear-sky extinction coefficient

    def setup_network(self, input_dim):
        """Setup neural network architecture with batch normalization."""
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.9, eps=1e-5),
            nn.Tanh(),
            PhysicsInformedLayer(128, 256),
            nn.BatchNorm1d(256, momentum=0.9, eps=1e-5),
            nn.Tanh(),
            PhysicsInformedLayer(256, 128),
            nn.BatchNorm1d(128, momentum=0.9, eps=1e-5),
            nn.Tanh(),
            PhysicsInformedLayer(128, 1)
        )

    def calculate_declination(self, day_of_year):
        """Calculate solar declination angle."""
        return torch.deg2rad(23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365))

    def calculate_hour_angle(self, time):
        """Calculate hour angle."""
        return torch.deg2rad(15 * (time - 12))  # Convert hour to radians

    def calculate_zenith_angle(self, lat, declination, hour_angle):
        """Calculate solar zenith angle."""
        lat_rad = torch.deg2rad(lat)
        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) + 
                     torch.cos(lat_rad) * torch.cos(declination) * torch.cos(hour_angle))
        return torch.arccos(torch.clamp(cos_zenith, -1, 1))

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten & Young formula."""
        zenith_deg = torch.rad2deg(zenith_angle)
        denominator = torch.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_deg) ** -1.6364
        return torch.where(zenith_deg < 90, 1 / denominator, float('inf'))

    def calculate_surface_orientation_factor(self, zenith_angle, slope, sun_azimuth, aspect):
        """Calculate surface orientation factor."""
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        sun_azimuth_rad = torch.deg2rad(sun_azimuth)
        
        return (torch.cos(slope_rad) * torch.cos(zenith_angle) + 
                torch.sin(slope_rad) * torch.sin(zenith_angle) * 
                torch.cos(sun_azimuth_rad - aspect_rad))

    def boundary_conditions(self, lat, day_of_year):
        """Calculate sunrise and sunset times."""
        declination = self.calculate_declination(day_of_year)
        lat_rad = torch.deg2rad(lat)
        
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(declination)
        cos_hour_angle = torch.clamp(cos_hour_angle, -1, 1)
        
        hour_angle = torch.arccos(cos_hour_angle)
        sunrise = 12 - torch.rad2deg(hour_angle) / 15
        sunset = 12 + torch.rad2deg(hour_angle) / 15
        
        return sunrise, sunset

    def forward(self, x):
        """Forward pass with essential physics constraints using normalized inputs."""
        # Handle single sample case for inference
        if x.size(0) == 1:
            self.eval()  # Set to evaluation mode for single sample
            for module in self.physics_net.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        
        # Extract normalized input components
        lat_norm, lon_norm, time_norm, slope_norm, aspect_norm = x.split(1, dim=1)
        
        # Denormalize inputs for physics calculations
        lat = lat_norm * 90
        lon = lon_norm * 180
        time = time_norm * 24
        slope = slope_norm * 180
        aspect = aspect_norm * 360
        
        # Calculate day of year from denormalized time
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Calculate solar position
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        zenith_angle = self.calculate_zenith_angle(lat, declination, hour_angle)
        
        # Calculate atmospheric effects
        air_mass = self.calculate_air_mass(zenith_angle)
        atmospheric_transmission = torch.exp(-self.atmospheric_extinction * air_mass)
        
        # Calculate surface orientation
        sun_azimuth = torch.rad2deg(hour_angle)  # Simplified sun azimuth
        surface_factor = self.calculate_surface_orientation_factor(zenith_angle, slope, sun_azimuth, aspect)
        
        # Neural network prediction (normalized)
        prediction = self.physics_net(x)
        
        # Re-dimensionalize the prediction and apply physical constraints
        prediction = prediction * self.solar_constant  # Convert normalized output back to W/m²
        constrained_prediction = (prediction * 
                               atmospheric_transmission * 
                               surface_factor)
        
        # Apply day/night boundary conditions
        sunrise, sunset = self.boundary_conditions(lat, day_of_year)
        is_daytime = (hour_of_day >= sunrise) & (hour_of_day <= sunset)
        
        final_prediction = torch.where(is_daytime, 
                                    torch.clamp(constrained_prediction, min=0.0),
                                    torch.zeros_like(constrained_prediction))
        
        # Return normalized prediction for training
        return final_prediction / self.solar_constant

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001, min_lr=0.0001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=min_lr
        )

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x_data)
        
        # Calculate data loss with adaptive weighting
        data_loss = F.mse_loss(y_pred, y_data)
        prediction_error = torch.abs(y_pred - y_data)
        adaptive_weight = torch.exp(-prediction_error)
        weighted_data_loss = torch.mean(data_loss * adaptive_weight)
        
        # Extract time components for boundary conditions
        lat = x_data[:, 0] * 90
        time = x_data[:, 2] * 24
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Enhanced physics constraints
        sunrise, sunset = self.model.boundary_conditions(lat, day_of_year)
        night_mask = (hour_of_day < sunrise) | (hour_of_day > sunset)
        physics_loss = torch.mean(y_pred[night_mask] ** 2)
        
        # Gradient penalty for physical consistency
        grad_outputs = torch.ones_like(y_pred)
        gradients = torch.autograd.grad(
            y_pred, x_data,
            grad_outputs=grad_outputs,
            create_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = torch.mean(gradients.pow(2))
        
        # Combined loss with enhanced physics constraints
        total_loss = (
            weighted_data_loss +
            0.1 * physics_loss +
            0.01 * gradient_penalty
        )
        
        # Backward pass with retain_graph
        total_loss.backward(retain_graph=True)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Clean up graphs
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
        
        return total_loss.item()
