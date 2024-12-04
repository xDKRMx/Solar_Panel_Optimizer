import numpy as np
import torch

class TerrainModel:
    def __init__(self):
        self.terrain_resolution = 100  # meters per grid point
        
    def compute_terrain_shading(self, x, solar_zenith, solar_azimuth):
        """
        Compute shading effects from surrounding terrain
        Args:
            x: Input tensor containing terrain elevation data
            solar_zenith: Solar zenith angle in radians
            solar_azimuth: Solar azimuth angle in radians
        Returns:
            Shading factor (0-1) where 0 is fully shaded
        """
        # Convert solar angles to direction vector
        sun_x = torch.sin(solar_zenith) * torch.cos(solar_azimuth)
        sun_y = torch.sin(solar_zenith) * torch.sin(solar_azimuth)
        sun_z = torch.cos(solar_zenith)
        sun_vector = torch.stack([sun_x, sun_y, sun_z], dim=-1)
        
        # Calculate terrain gradients
        dx = torch.gradient(x[..., 2], dim=1)[0] / self.terrain_resolution
        dy = torch.gradient(x[..., 2], dim=0)[0] / self.terrain_resolution
        
        # Surface normal vectors
        normal = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=-1)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        
        # Compute cosine of incident angle
        cos_theta = torch.sum(normal * sun_vector, dim=-1)
        
        # Calculate shading factor
        shading = torch.clamp(cos_theta, min=0.0)
        
        return shading

    def compute_diffuse_radiation(self, x, temperature, pressure):
        """
        Compute diffuse radiation based on atmospheric conditions
        Args:
            x: Input tensor containing position data
            temperature: Air temperature in Kelvin
            pressure: Atmospheric pressure in Pa
        Returns:
            Diffuse radiation factor
        """
        # Approximate atmospheric turbidity using temperature and pressure
        turbidity = 2.0 + 0.1 * (temperature - 288.15) / 10.0
        
        # Altitude-based air mass calculation
        altitude = x[..., 2]
        air_mass = pressure / 101325.0 * torch.exp(-altitude / 8500.0)
        
        # Diffuse radiation model (simplified Perez model)
        diffuse_factor = 1.0 - torch.exp(-turbidity * air_mass)
        
        return diffuse_factor
