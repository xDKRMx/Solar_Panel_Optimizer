import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def boundary_conditions(self, x):
        """Physical boundary conditions"""
        # Unpack parameters
        lat, lon, time, slope, aspect, atm, cloud, wavelength = x.split(1, dim=1)
        
        # Top of atmosphere condition (solar constant)
        # Placeholder for zenith angle calculation - needs implementation
        cos_zenith = torch.ones_like(lat) # Placeholder, needs proper calculation.
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



class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate
        )

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model(x_data)

        # Data loss
        data_loss = F.mse_loss(y_pred, y_data)

        # Physics-informed loss
        physics_loss = self.model.physics_loss(x_data, y_pred)

        # Boundary condition loss
        toa_condition, _ = self.model.boundary_conditions(x_data)
        bc_loss = F.mse_loss(y_pred, toa_condition) #Using MSE loss for BC comparison

        # Total loss with adaptive weights
        w1, w2, w3 = self.calculate_adaptive_weights(
            data_loss, physics_loss, bc_loss
        )
        total_loss = w1*data_loss + w2*physics_loss + w3*bc_loss

        # Backward pass with physics constraints
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def calculate_adaptive_weights(self, data_loss, physics_loss, bc_loss):
        # Normalize losses
        total_magnitude = data_loss + physics_loss + bc_loss
        if total_magnitude == 0:  #Handle the case where all losses are zero to avoid division by zero.
            return 1/3, 1/3, 1/3
        w1 = physics_loss / total_magnitude
        w2 = data_loss / total_magnitude
        w3 = bc_loss / total_magnitude
        return w1, w2, w3