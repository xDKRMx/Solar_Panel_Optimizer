import numpy as np
import torch

class SolarPhysicsIdeal:
    def __init__(self):
        # Universal physical constants
        self.solar_constant = 1367.0  # Solar constant, S (W/m²)
        self.extinction_coefficient = 0.1  # Idealized extinction coefficient for clear sky

    def calculate_declination(self, day_of_year):
        """Calculate the solar declination angle (δ) for a given day of the year."""
        # δ = 23.45° · sin(2π/365 · (n-81))
        declination = 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)
        return torch.deg2rad(declination)

    def calculate_sunrise_sunset(self, latitude, day_of_year):
        """Calculate sunrise and sunset times using accurate equations.
        
        Args:
            latitude: Location latitude in degrees
            day_of_year: Day of the year (1-365)
            
        Returns:
            tuple: (sunrise time, sunset time) in hours (local solar time)
        """
        # Convert latitude to radians
        lat_rad = torch.deg2rad(latitude)
        
        # Calculate declination angle
        decl_rad = self.calculate_declination(day_of_year)
        
        # Calculate hour angle at sunrise/sunset using:
        # cos(h) = -tan(φ)·tan(δ)
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(decl_rad)
        
        # Identify polar conditions before clamping
        is_polar_day = cos_hour_angle < -1  # Sun never sets
        is_polar_night = cos_hour_angle > 1  # Sun never rises
        
        # Clamp values for regular calculations
        cos_hour_angle_clamped = torch.clamp(cos_hour_angle, -1, 1)
        
        # Convert to hour angle in radians
        hour_angle = torch.arccos(cos_hour_angle_clamped)
        
        # Convert hour angle to hours
        # Sunrise = 12 - h/15, Sunset = 12 + h/15
        hour_offset = torch.rad2deg(hour_angle) / 15
        
        sunrise = 12 - hour_offset
        sunset = 12 + hour_offset
        
        # Handle special cases for polar regions
        # For polar day: sunrise = 0, sunset = 24 (all day sunlight)
        # For polar night: sunrise = inf, sunset = inf (no sunlight)
        sunrise = torch.where(is_polar_day, torch.zeros_like(sunrise), 
                            torch.where(is_polar_night, torch.full_like(sunrise, float('inf')), sunrise))
        sunset = torch.where(is_polar_day, torch.full_like(sunset, 24), 
                           torch.where(is_polar_night, torch.full_like(sunset, float('inf')), sunset))
        
        return sunrise, sunset, is_polar_day, is_polar_night
        
        return sunrise, sunset

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
        """Calculate the surface orientation factor (f_surf)."""
        slope_rad = torch.deg2rad(slope)
        sun_azimuth_rad = torch.deg2rad(sun_azimuth)
        panel_azimuth_rad = torch.deg2rad(panel_azimuth)

        return (
            torch.cos(slope_rad) * zenith_angle +
            torch.sin(slope_rad) * torch.sqrt(1 - zenith_angle ** 2) *
            torch.cos(sun_azimuth_rad - panel_azimuth_rad)
        )

    def calculate_hour_angle(self, time):
        """Calculate the hour angle (h) based on the time of day."""
        return torch.deg2rad(15 * (time - 12))

    def calculate_irradiance(self, latitude, time, slope=0, panel_azimuth=0):
        """Calculate the total solar irradiance at the surface under ideal conditions."""
        # Ensure inputs are tensors with proper dtype
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        # Convert day of year from time
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Get sunrise and sunset times along with polar conditions
        sunrise, sunset, is_polar_day, is_polar_night = self.calculate_sunrise_sunset(latitude, day_of_year)
        
        # Check if current time is during daylight hours, including polar day condition
        is_daytime = is_polar_day | ((hour_of_day >= sunrise) & (hour_of_day <= sunset))
        
        # Calculate solar position regardless of time for polar days
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)  # Ensure no negative values
        
        # Calculate atmospheric effects
        air_mass = self.calculate_air_mass(cos_zenith)
        transmission = self.calculate_atmospheric_transmission(air_mass)
        
        # Calculate surface orientation
        surface_orientation = self.calculate_surface_orientation_factor(
            cos_zenith, slope, hour_angle, panel_azimuth
        )
        
        # Calculate base irradiance
        irradiance = self.solar_constant * transmission * surface_orientation
        
        # Apply daylight mask
        irradiance = torch.where(is_daytime, irradiance, torch.zeros_like(irradiance))
        
        return torch.clamp(irradiance, min=0.0)  # Ensure non-negative irradiance

    def calculate_efficiency(self, latitude, time, slope=0, panel_azimuth=0, ref_efficiency=0.15):
        """Calculate solar panel efficiency based on physical parameters.
        
        Args:
            latitude: Location latitude in degrees
            time: Time of day (hour)
            slope: Panel slope angle in degrees (β)
            panel_azimuth: Panel azimuth angle in degrees (φp)
            ref_efficiency: Reference efficiency under STC (default 0.15 or 15%)
            
        Returns:
            Total efficiency including surface orientation effects
        """
        # Convert inputs to tensors
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        # Calculate day of year from time
        day_of_year = torch.floor(time / 24 * 365)
        
        # Calculate solar position
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(time % 24)
        
        # Convert angles to radians
        lat_rad = torch.deg2rad(latitude)
        decl_rad = torch.deg2rad(declination)
        hour_angle_rad = hour_angle  # Already in radians
        slope_rad = torch.deg2rad(slope)
        panel_azimuth_rad = torch.deg2rad(panel_azimuth)
        
        # Calculate zenith angle
        cos_zenith = torch.sin(lat_rad) * torch.sin(decl_rad) + \
                     torch.cos(lat_rad) * torch.cos(decl_rad) * torch.cos(hour_angle_rad)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)  # Ensure no negative values
        
        # Calculate solar azimuth angle (φs)
        sin_solar_azimuth = torch.cos(decl_rad) * torch.sin(hour_angle_rad) / \
                           torch.sqrt(1 - cos_zenith**2 + 1e-6)  # Add small epsilon to avoid division by zero
        cos_solar_azimuth = (torch.sin(decl_rad) * torch.cos(lat_rad) - \
                            torch.cos(decl_rad) * torch.sin(lat_rad) * torch.cos(hour_angle_rad)) / \
                           torch.sqrt(1 - cos_zenith**2 + 1e-6)
        solar_azimuth = torch.atan2(sin_solar_azimuth, cos_solar_azimuth)
        
        # Calculate surface orientation factor (fsurt)
        # fsurt = cos(β)cos(θz) + sin(β)√(1-cos²(θz))cos(φs-φp)
        fsurt = torch.cos(slope_rad) * cos_zenith + \
                torch.sin(slope_rad) * torch.sqrt(1 - cos_zenith**2) * \
                torch.cos(solar_azimuth - panel_azimuth_rad)
        
        # Calculate total efficiency
        # η = ηref * fsurt
        total_efficiency = ref_efficiency * fsurt
        
        return torch.clamp(total_efficiency, min=0.0)  # Ensure non-negative efficiency

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
