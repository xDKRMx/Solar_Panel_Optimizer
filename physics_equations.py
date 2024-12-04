import jax.numpy as jnp
from jax import grad, jit

class SolarRadiationPhysics:
    def __init__(self):
        # Solar constant (W/m^2)
        self.solar_constant = 1367.0
        # Stefan-Boltzmann constant (W/(m^2*K^4))
        self.stefan_boltzmann = 5.67e-8

    def radiative_transfer_equation(self, x, t, I):
        """
        Radiative transfer equation:
        dI/dt + c∂I/∂x = -κI + η
        
        I: radiation intensity
        κ: absorption coefficient
        η: emission coefficient
        """
        dI_dt = grad(lambda t: I(t, x))(t)
        dI_dx = grad(lambda x: I(t, x))(x)

        # Absorption and emission terms
        kappa = self.compute_absorption_coefficient(x)
        eta = self.compute_emission_coefficient(x)

        return dI_dt + 3e8 * dI_dx + kappa * I(x, t) - eta

    def compute_absorption_coefficient(self, x):
        # Simplified absorption coefficient based on atmospheric density
        return 0.1 * jnp.exp(-x / 10000)  # x is height in meters

    def compute_emission_coefficient(self, x):
        # Simplified emission coefficient
        return 0.05 * jnp.exp(-x / 15000)

    def solar_position_equations(self, latitude, longitude, time):
        """
        Solar position equations (zenith and azimuth angles)
        """
        # Convert time to hour angle
        hour_angle = 15.0 * (time - 12.0) * (jnp.pi/180.0)
        lat_rad = latitude * (jnp.pi/180.0)
        
        # Solar declination
        day_angle = 2 * jnp.pi * time / 365.25
        declination = 0.396372 - 22.91327 * jnp.cos(day_angle) + 4.02543 * jnp.sin(day_angle)
        declination_rad = declination * (jnp.pi/180.0)
        
        # Calculate zenith angle
        cos_zenith = (
            jnp.sin(lat_rad) * jnp.sin(declination_rad) +
            jnp.cos(lat_rad) * jnp.cos(declination_rad) * jnp.cos(hour_angle)
        )
        
        return jnp.arccos(cos_zenith)
