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
        super().__init__()
        self.setup_physical_constants()
        self.setup_network(input_dim)
        
        # Initialize terrain model
        from .terrain_model import TerrainModel
        self.terrain_model = TerrainModel()
    
    def setup_physical_constants(self):
        # Physical constants
        self.solar_constant = 1367.0
        self.stefan_boltzmann = 5.67e-8
        self.atmospheric_extinction = 0.1
        self.earth_radius = 6371000
        self.air_mass_coefficient = 1.0
        
        # Additional physics parameters
        self.beta = 0.1  # Aerosol optical thickness
        self.alpha = 1.3  # Ångström exponent
        self.cloud_alpha = 0.75  # Cloud transmission parameter
        self.albedo = 0.2  # Ground reflectance
    
    def setup_network(self, input_dim):
        """Setup neural network architecture"""
        self.physics_net = nn.Sequential(
            PhysicsInformedLayer(input_dim, 128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 256),
            nn.Tanh(),
            PhysicsInformedLayer(256, 128),
            nn.Tanh(),
            PhysicsInformedLayer(128, 1)
        )

    def forward(self, x):
        prediction = self.physics_net(x)
        prediction = self.apply_physical_constraints(x, prediction)
        return prediction
        
    def apply_physical_constraints(self, x, prediction):
        """Apply physical constraints to predictions"""
        lat, lon, time, slope, aspect, atm, cloud, wavelength = x.split(1, dim=1)
        max_possible = self.calculate_max_possible_irradiance(lat, time)
        atmospheric_attenuation = self.calculate_atmospheric_attenuation(atm, cloud)
        surface_factor = self.calculate_surface_orientation_factor(lat, lon, time, slope, aspect)
        physically_constrained = torch.min(prediction, max_possible) * atmospheric_attenuation * surface_factor
        return physically_constrained
        
    def calculate_max_possible_irradiance(self, lat, time):
        """Calculate maximum possible irradiance based on solar position"""
        declination = 23.45 * torch.sin(2 * torch.pi * (time - 81) / 365)
        cos_zenith = torch.sin(torch.deg2rad(lat)) * torch.sin(torch.deg2rad(declination)) + \
                     torch.cos(torch.deg2rad(lat)) * torch.cos(torch.deg2rad(declination))
        return self.solar_constant * torch.clamp(cos_zenith, min=0.0)
    
    def calculate_atmospheric_attenuation(self, atm, cloud):
        """Calculate atmospheric attenuation factor"""
        return atm * (1.0 - self.cloud_alpha * cloud**3)
    
    def calculate_surface_orientation_factor(self, lat, lon, time, slope, aspect):
        """Calculate surface orientation factor"""
        # Convert angles to radians
        lat_rad = torch.deg2rad(lat)
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)
        
        # Solar position calculations
        hour_angle = 2 * torch.pi * (time / 24 - 0.5)
        declination = 23.45 * torch.sin(2 * torch.pi * (time - 81) / 365)
        declination_rad = torch.deg2rad(declination)
        
        # Calculate solar zenith and azimuth angles
        cos_zenith = torch.sin(lat_rad) * torch.sin(declination_rad) + \
                     torch.cos(lat_rad) * torch.cos(declination_rad) * torch.cos(hour_angle)
        
        # Calculate surface factor
        surface_factor = torch.cos(slope_rad) * cos_zenith + \
                        torch.sin(slope_rad) * torch.sqrt(1 - cos_zenith**2) * \
                        torch.cos(aspect_rad - torch.atan2(torch.sin(hour_angle),
                                torch.cos(hour_angle) * torch.sin(lat_rad) -
                                torch.tan(declination_rad) * torch.cos(lat_rad)))
        
        return torch.clamp(surface_factor, min=0.0)
    
    def radiative_transfer_equation(self, x, I):
        """Core PDE: Radiative Transfer Equation"""
        # Get gradients
        I_gradients = torch.autograd.grad(I, x, create_graph=True)[0]
        extinction = self.calculate_extinction(x)
        emission = self.calculate_emission(x)
        scattering = self.calculate_scattering(x, I)
        
        # Complete RTE equation
        rte_residual = I_gradients + extinction*I - emission - scattering
        return rte_residual
    
    def calculate_extinction(self, x):
        """Calculate extinction coefficient"""
        _, _, _, _, _, atm, cloud, wavelength = x.split(1, dim=1)
        return self.beta * (wavelength)**(-self.alpha) * atm * (1 - self.cloud_alpha * cloud**3)

    def boundary_conditions(self, x):
        """Physical boundary conditions"""
        lat, lon, time, slope, aspect, atm, cloud, wavelength = x.split(1, dim=1)
        # Top of atmosphere condition
        toa_condition = self.solar_constant * self.calculate_zenith_factor(lat, time)
        # Surface condition with all components
        surface_condition = (
            self.calculate_surface_emission(x) +
            self.calculate_surface_reflection(x) +
            self.calculate_diffuse_radiation(x)
        )
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
        # RTE loss
        rte_loss = self.radiative_transfer_equation(x, y_pred)
        
        # Boundary conditions
        boundary_loss = self.boundary_conditions(x)
        
        # Conservation laws
        conservation_loss = self.energy_conservation(x, y_pred)
        
        # Spectral constraints
        spectral_loss = self.spectral_constraints(x, y_pred)
        
        # Combined physics loss
        return (rte_loss + boundary_loss + 
                conservation_loss + spectral_loss)
                
    def spectral_constraints(self, x, y_pred):
        """Apply spectral physics constraints"""
        wavelength = x.split(1, dim=1)[-1]
        spectral_data = self.data_processor.prepare_spectral_data(x)
        return torch.mean((y_pred - spectral_data)**2)
        
    def energy_conservation(self, x, y_pred):
        """Energy conservation constraint"""
        incoming = self.calculate_incoming_radiation(x)
        outgoing = self.calculate_outgoing_radiation(x, y_pred)
        absorbed = self.calculate_absorbed_radiation(x, y_pred)
        
        return torch.mean(torch.abs(incoming - (outgoing + absorbed)))



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

    def calculate_incoming_radiation(self, x):
        """Calculate incoming solar radiation"""
        lat, _, time, _, _, atm, cloud, _ = x.split(1, dim=1)
        zenith_factor = self.calculate_zenith_factor(lat, time)
        atmospheric_attenuation = self.calculate_atmospheric_attenuation(atm, cloud)
        return self.solar_constant * zenith_factor * atmospheric_attenuation
    
    def calculate_outgoing_radiation(self, x, y_pred):
        """Calculate outgoing radiation"""
        # Consider both reflection and emission
        surface_reflection = self.albedo * y_pred
        surface_emission = self.calculate_emission(x)
        return surface_reflection + surface_emission
    
    def calculate_absorbed_radiation(self, x, y_pred):
        """Calculate absorbed radiation"""
        return (1 - self.albedo) * y_pred
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