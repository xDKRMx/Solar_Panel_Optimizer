import numpy as np
import torch
import torch.nn as nn


class PhysicsInformedLayer(nn.Module):
    """Custom layer with physics constraints"""
    def __init__(self, in_features, out_features):
        super(PhysicsInformedLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.physics_weights = nn.Parameter(
            torch.randn(out_features)
        )
    
    def forward(self, x):
        # Linear transformation
        out = self.linear(x)
        # Apply physics-based transformation
        out = out * torch.sigmoid(self.physics_weights)
        return out


class SolarPINN(nn.Module):
    def __init__(self, input_dim=8):
        super(SolarPINN, self).__init__()
        # Physics-informed neural network architecture
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 256),
            nn.Tanh(),
            PhysicsInformedLayer(256, 1)
        )
        
        # Physical constants
        self.I0 = 1367.0  # Solar constant (W/m²)
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant
        self.c = 3e8  # Speed of light (m/s)
        self.h = 6.626e-34  # Planck constant
        self.kb = 1.380649e-23  # Boltzmann constant
        
        # Reference values
        self.T_ref = 298.15  # Reference temperature (K)
        self.lambda_ref = 0.5e-6  # Reference wavelength (m)
        self.time_ref = 24.0  # Time reference (hours)
        
        # RTE parameters
        self.extinction_coeff = 0.1  # Base extinction coefficient
        self.scattering_albedo = 0.9  # Single scattering albedo
        self.asymmetry_factor = 0.85  # Scattering asymmetry factor

    def forward(self, x):
        return self.physics_net(x)

    def radiative_transfer_equation(self, x, y_pred):
        """Implement core RTE with automatic differentiation"""
        # Extract spatial coordinates for gradient calculation
        lat, lon = x[:, 0], x[:, 1]
        
        # Calculate spatial gradients using autograd
        I_gradients = torch.autograd.grad(
            y_pred,
            x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        
        # RTE components calculation
        extinction = self.calculate_extinction(x)
        emission = self.calculate_emission(x)
        scattering = self.calculate_scattering(x, y_pred)
        
        # Complete RTE: dI/ds = -extinction*I + emission + scattering
        rte_residual = I_gradients + extinction*y_pred - emission - scattering
        
        return rte_residual

    def calculate_extinction(self, x):
        """Calculate extinction coefficient based on atmospheric conditions"""
        _, _, _, _, _, atm, cloud_cover, wavelength = (x[:, i] for i in range(8))
        wavelength_scaled = wavelength / self.lambda_ref
        
        # Wavelength-dependent extinction
        extinction = self.extinction_coeff * (wavelength_scaled**-1.3)
        
        # Add cloud effects
        extinction = extinction * (1 + 2*cloud_cover)
        
        # Atmospheric transmission effects
        extinction = extinction * (2 - atm)
        
        return extinction

    def calculate_emission(self, x):
        """Calculate thermal emission term"""
        # Simplified blackbody emission at reference temperature
        _, _, _, _, _, atm, _, wavelength = (x[:, i] for i in range(8))
        wavelength_m = wavelength * 1e-6  # Convert to meters
        
        # Planck's law
        exp_term = torch.exp(self.h*self.c/(wavelength_m*self.kb*self.T_ref))
        emission = 2*self.h*self.c**2/(wavelength_m**5 * (exp_term - 1))
        
        # Scale by atmospheric transmission
        emission = emission * atm
        
        return emission

    def calculate_scattering(self, x, I):
        """Calculate scattering contribution using Henyey-Greenstein phase function"""
        # Extract relevant parameters
        _, _, _, _, _, _, cloud_cover, _ = (x[:, i] for i in range(8))
        
        # Modify scattering based on cloud cover
        scattering_coeff = self.scattering_albedo * (1 + cloud_cover)
        
        # Simplified phase function integral
        phase_integral = 0.5 * (1 - self.asymmetry_factor**2) / \
                        (1 + self.asymmetry_factor**2)**1.5
        
        return scattering_coeff * phase_integral * I

    def boundary_conditions(self, x):
        """Implement physical boundary conditions"""
        # TOA condition (top of atmosphere)
        toa_condition = self.calculate_toa_condition(x)
        
        # Surface boundary condition
        surface_condition = self.calculate_surface_condition(x)
        
        return toa_condition, surface_condition

    def calculate_toa_condition(self, x):
        """Top of atmosphere boundary condition"""
        _, _, time, _, _, _, _, _ = (x[:, i] for i in range(8))
        
        # Solar zenith angle dependent condition
        cos_zenith = torch.cos(2*np.pi*time/24)  # Simplified solar position
        toa_flux = self.I0 * torch.clamp(cos_zenith, min=0)
        
        return toa_flux

    def calculate_surface_condition(self, x):
        """Surface boundary condition with ground reflection"""
        _, _, _, slope, _, _, _, _ = (x[:, i] for i in range(8))
        
        # Surface albedo dependent on slope
        surface_albedo = 0.2 + 0.1 * torch.sin(torch.deg2rad(slope))
        
        return surface_albedo

    def physics_loss(self, x, y_pred):
        """Complete physics-informed loss calculation"""
        # RTE residual
        rte_residual = self.radiative_transfer_equation(x, y_pred)
        
        # Boundary conditions
        toa_loss, surface_loss = self.boundary_conditions(x)
        
        # Energy conservation check
        energy_conservation = self.check_energy_conservation(x, y_pred)
        
        # Combine losses with adaptive weights
        return torch.mean(
            torch.abs(rte_residual)**2 +
            torch.abs(toa_loss)**2 +
            torch.abs(surface_loss)**2 +
            torch.abs(energy_conservation)**2
        )

    def check_energy_conservation(self, x, y_pred):
        """Verify energy conservation in the system"""
        # Extract relevant parameters
        _, _, _, _, _, atm, cloud_cover, _ = (x[:, i] for i in range(8))
        
        # Calculate energy fluxes
        incident_flux = self.I0 * atm
        absorbed_flux = y_pred * (1 - cloud_cover)
        scattered_flux = y_pred * cloud_cover * self.scattering_albedo
        
        # Energy conservation residual
        return incident_flux - (absorbed_flux + scattered_flux)


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
        
        # Forward pass
        y_pred = self.model(x_data)
        
        # Calculate losses
        data_loss = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        
        # Adaptive weighting
        loss_ratio = data_loss.item() / (physics_loss.item() + 1e-8)
        alpha = torch.sigmoid(torch.tensor(loss_ratio))
        
        # Total loss
        total_loss = alpha * data_loss + (1 - alpha) * physics_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)
        
        return total_loss.item(), data_loss.item(), physics_loss.item()