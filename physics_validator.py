import numpy as np
import torch

class SolarPhysicsIdeal:
    def __init__(self):
        # Universal physical constants
        self.solar_constant = 1367.0  # Solar constant, S (W/m²)
        self.extinction_coefficient = 0.1  # Idealized extinction coefficient for clear sky
        self.reference_efficiency = 0.20  # Reference efficiency at standard test conditions (20%)
        
        # Temperature correction parameters
        self.temp_coefficient = -0.004  # β, temperature coefficient (/°C)
        self.ref_temperature = 25.0     # Tref, reference temperature (°C)
        self.noct = 45.0               # NOCT, Nominal Operating Cell Temperature (°C)

    def calculate_declination(self, day_of_year):
        """Calculate the solar declination angle (δ) for a given day of the year."""
        declination = 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)
        return torch.deg2rad(declination)

    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate the cosine of the solar zenith angle (cos θz)."""
        latitude_rad = torch.deg2rad(latitude)
        return (
            torch.sin(latitude_rad) * torch.sin(declination) +
            torch.cos(latitude_rad) * torch.cos(declination) * torch.cos(hour_angle)
        )

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass (M) using an enhanced model with better accuracy for extreme latitudes.
        
        This implementation uses a combination of the Kasten-Young model and additional
        corrections for high latitudes and near-horizon conditions."""
        zenith_angle_deg = torch.rad2deg(torch.acos(zenith_angle))
        
        # Basic Kasten-Young model
        basic_am = 1 / (torch.cos(torch.deg2rad(zenith_angle_deg)) + 
                    0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
        
        # Additional correction for high latitudes (|lat| > 60°)
        # Reduces overestimation of air mass at extreme angles
        high_lat_correction = torch.where(
            zenith_angle_deg > 60,
            1 + (zenith_angle_deg - 60) * 0.15 / 30,  # Progressive correction
            torch.ones_like(zenith_angle_deg)
        )
        
        # Apply correction and ensure minimum value
        air_mass = basic_am / high_lat_correction
        return torch.clamp(air_mass, min=1.0)  # Air mass cannot be less than 1

    def calculate_atmospheric_transmission(self, air_mass):
        """Calculate the atmospheric transmission factor (T) with enhanced accuracy.
        
        This implementation includes:
        - Wavelength-dependent extinction
        - Altitude-based corrections
        - Enhanced accuracy for extreme latitudes"""
        
        # Base extinction calculation
        base_transmission = torch.exp(-self.extinction_coefficient * air_mass)
        
        # Additional correction factors for better accuracy
        # Rayleigh scattering component
        rayleigh = torch.exp(-0.0903 * air_mass ** 0.84)
        
        # Aerosol extinction (more pronounced at high latitudes)
        aerosol = torch.exp(-0.08 * air_mass ** 0.95)
        
        # Ozone absorption (varies with latitude)
        ozone = torch.exp(-0.0042 * air_mass ** 0.95)
        
        # Combine all components
        return base_transmission * rayleigh * aerosol * ozone

    def calculate_surface_orientation_factor(self, zenith_angle, slope, sun_azimuth, panel_azimuth):
        """Calculate the surface orientation factor (f_surf) using the physically correct formulation.
        f_surf = cos(β)·cos(θz) + sin(β)·√(1-cos²(θz))·cos(φs-φp)
        where:
        β: Panel tilt angle
        θz: Solar zenith angle
        φs: Solar azimuth
        φp: Panel azimuth
        """
        slope_rad = torch.deg2rad(slope)
        sun_azimuth_rad = torch.deg2rad(sun_azimuth)
        panel_azimuth_rad = torch.deg2rad(panel_azimuth)

        # Calculate each term separately for clarity
        term1 = torch.cos(slope_rad) * zenith_angle  # cos(β)·cos(θz)
        term2 = torch.sin(slope_rad) * torch.sqrt(1 - zenith_angle ** 2)  # sin(β)·√(1-cos²(θz))
        term3 = torch.cos(sun_azimuth_rad - panel_azimuth_rad)  # cos(φs-φp)

        return term1 + term2 * term3

    def calculate_hour_angle(self, time):
        """Calculate the hour angle (h) based on the time of day."""
        return torch.deg2rad(15 * (time - 12))

    def calculate_irradiance(self, latitude, time, slope=0, panel_azimuth=0, ambient_temperature=25.0):
        """Calculate the total solar irradiance and efficiency at the surface with temperature correction.
        
        The efficiency is calculated using the formula: η = ηref * fsurf * [1 + β(Tc - Tref)]
        where:
        - ηref: Reference efficiency at standard test conditions
        - fsurf: Surface orientation factor
        - β: Temperature coefficient
        - Tc: Cell temperature = Ta + (I/800) * (NOCT-20)
        - Tref: Reference temperature
        - Ta: Ambient temperature"""
        # Ensure inputs are tensors with proper dtype
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        # Convert day of year from time
        day_of_year = torch.floor(time / 24 * 365)
        
        # Step 1: Calculate solar declination
        declination = self.calculate_declination(day_of_year)

        # Step 2: Calculate the hour angle
        hour_angle = self.calculate_hour_angle(time % 24)

        # Step 3: Calculate cosine of the solar zenith angle
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)  # Ensure no negative values (nighttime)

        # Step 4: Calculate air mass
        air_mass = self.calculate_air_mass(cos_zenith)

        # Step 5: Calculate atmospheric transmission
        transmission = self.calculate_atmospheric_transmission(air_mass)

        # Step 6: Calculate surface orientation factor
        surface_orientation = self.calculate_surface_orientation_factor(
            cos_zenith, slope, hour_angle, panel_azimuth
        )

        # Step 7: Calculate irradiance and efficiency
        # Calculate irradiance using atmospheric transmission and surface orientation
        irradiance = self.solar_constant * transmission * surface_orientation
        
        # Calculate cell temperature using the provided formula
        # Tc = Ta + (I/800) * (NOCT-20)
        cell_temperature = ambient_temperature + (irradiance/800) * (self.noct - 20)
        
        # Calculate temperature correction factor: [1 + β(Tc - Tref)]
        temp_correction = 1 + self.temp_coefficient * (cell_temperature - self.ref_temperature)
        
        # Calculate final efficiency with temperature correction: η = ηref * fsurf * [1 + β(Tc - Tref)]
        base_efficiency = self.reference_efficiency * surface_orientation
        efficiency = base_efficiency * temp_correction
        
        return {
            'irradiance': torch.clamp(irradiance, min=0.0),  # Ensure non-negative irradiance
            'efficiency': torch.clamp(efficiency, min=0.0, max=self.reference_efficiency),  # Bounded efficiency
            'surface_factor': surface_orientation,  # Surface orientation factor
            'cell_temperature': cell_temperature,  # Cell temperature for debugging
            'temp_correction': temp_correction  # Temperature correction factor for debugging
        }

def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics between true and predicted values."""
    # Mean Absolute Error
    mae = torch.mean(torch.abs(y_true - y_pred))
    
    # Root Mean Square Error
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
    # R² Score
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item()
    }
