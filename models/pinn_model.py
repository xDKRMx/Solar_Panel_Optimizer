import torch
import torch.nn as nn
import numpy as np

class PhysicsInitializer:
    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            # Initialize based on input/output dimensions
            fan_in = m.weight.size(1)
            # He initialization scaled by solar constant
            std = np.sqrt(2.0 / fan_in) * (1367.0 / fan_in)  # 1367 W/m² is solar constant
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class SolarPINN(nn.Module):
    def __init__(self, input_dim=6):
        super(SolarPINN, self).__init__()
        
        # Define network architecture with batch normalization
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure non-negative predictions
        )
        
        # Initialize weights using physics-informed initialization
        self.apply(PhysicsInitializer.initialize_weights)
        
    def forward(self, x):
        return self.net(x)
    
    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm = (
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        )
        
        # Compute gradients for physics residual
        y_grad = torch.autograd.grad(
            y_pred.sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        
        # Enhanced physics residual incorporating solar position
        # and atmospheric effects
        solar_zenith = torch.cos(torch.deg2rad(lat)) * torch.cos(torch.deg2rad(15 * (time - 12)))
        
        # Physics residual based on advection-diffusion equation
        # with enhanced solar position dependency
        residual = (y_grad[:, 2] +  # Time derivative
                   y_grad[:, 0] * solar_zenith +  # Latitude effect
                   y_grad[:, 1] +  # Longitude effect
                   (1 - atm) * y_pred)  # Atmospheric attenuation
        
        return torch.mean(residual ** 2)

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x_data)
        
        # Compute losses
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        
        # Increased physics loss weight (from 0.1 to 0.5)
        total_loss = mse + 0.5 * physics_loss
        
        # Add regularization for physical bounds
        bounds_loss = torch.mean(torch.relu(-y_pred))  # Penalty for negative values
        total_loss += 0.1 * bounds_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mse.item(), physics_loss.item()
