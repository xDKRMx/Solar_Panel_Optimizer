import numpy as np
import torch

class SolarPhysicsIdeal:
    def __init__(self):
        # Universal physical constants
        self.solar_constant = 1367.0  # Solar constant, S (W/m²)
        self.extinction_coefficient = 0.1  # Idealized extinction coefficient for clear sky
        
        # Temperature-related constants
        self.temp_coefficient = -0.004  # Temperature coefficient (β), typically -0.004 K^-1
        self.ref_temperature = 25.0  # Reference temperature (Tref) at STC, °C
        self.noct = 45.0  # Nominal Operating Cell Temperature, °C

    def calculate_declination(self, day_of_year):
        """Calculate the solar declination angle (δ) for a given day of the year."""
        declination = 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)
        return torch.deg2rad(declination)

    def calculate_sunrise_sunset(self, latitude, day_of_year):
        """Calculate sunrise and sunset times using accurate equations."""
        lat_rad = torch.deg2rad(latitude)
        decl_rad = self.calculate_declination(day_of_year)
        cos_hour_angle = -torch.tan(lat_rad) * torch.tan(decl_rad)
        
        is_polar_day = cos_hour_angle < -1
        is_polar_night = cos_hour_angle > 1
        
        cos_hour_angle_clamped = torch.clamp(cos_hour_angle, -1, 1)
        hour_angle = torch.arccos(cos_hour_angle_clamped)
        hour_offset = torch.rad2deg(hour_angle) / 15
        
        sunrise = 12 - hour_offset
        sunset = 12 + hour_offset
        
        sunrise = torch.where(is_polar_day, torch.zeros_like(sunrise),
                            torch.where(is_polar_night, torch.full_like(sunrise, float('inf')), sunrise))
        sunset = torch.where(is_polar_day, torch.full_like(sunset, 24),
                           torch.where(is_polar_night, torch.full_like(sunset, float('inf')), sunset))
        
        return sunrise, sunset, is_polar_day, is_polar_night

    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate the cosine of the solar zenith angle (cos θz)."""
        latitude_rad = torch.deg2rad(latitude)
        return (
            torch.sin(latitude_rad) * torch.sin(declination) +
            torch.cos(latitude_rad) * torch.cos(declination) * torch.cos(hour_angle)
        )

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass (M) using an enhanced model."""
        zenith_angle_deg = torch.rad2deg(torch.acos(zenith_angle))
        basic_am = 1 / (torch.cos(torch.deg2rad(zenith_angle_deg)) +
                       0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
        high_lat_correction = torch.where(
            zenith_angle_deg > 60,
            1 + (zenith_angle_deg - 60) * 0.15 / 30,
            torch.ones_like(zenith_angle_deg)
        )
        air_mass = basic_am / high_lat_correction
        return torch.clamp(air_mass, min=1.0)

    def calculate_atmospheric_transmission(self, air_mass):
        """Calculate the atmospheric transmission factor (T)."""
        base_transmission = torch.exp(-self.extinction_coefficient * air_mass)
        rayleigh = torch.exp(-0.0903 * air_mass ** 0.84)
        aerosol = torch.exp(-0.08 * air_mass ** 0.95)
        ozone = torch.exp(-0.0042 * air_mass ** 0.95)
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

    def calculate_cell_temperature(self, ambient_temp, irradiance):
        """Calculate cell temperature using NOCT method.
        
        Args:
            ambient_temp: Ambient temperature in °C
            irradiance: Solar irradiance in W/m²
            
        Returns:
            Cell temperature in °C
        """
        return ambient_temp + (irradiance / 800) * (self.noct - 20)

    def calculate_efficiency(self, latitude, time, slope=0, panel_azimuth=0, ref_efficiency=0.15, ambient_temp=25):
        """Calculate temperature-corrected solar panel efficiency.
        
        Args:
            latitude: Location latitude in degrees
            time: Time of day (hour)
            slope: Panel slope angle in degrees (β)
            panel_azimuth: Panel azimuth angle in degrees (φp)
            ref_efficiency: Reference efficiency under STC (default 0.15 or 15%)
            ambient_temp: Ambient temperature in °C (default 25°C)
            
        Returns:
            Temperature-corrected total efficiency
        """
        # Convert inputs to tensors
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        ambient_temp = torch.as_tensor(ambient_temp, dtype=torch.float32)
        
        # Calculate day of year and time components
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Calculate solar position
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        
        # Calculate zenith angle
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)
        
        # Calculate irradiance
        air_mass = self.calculate_air_mass(cos_zenith)
        transmission = self.calculate_atmospheric_transmission(air_mass)
        irradiance = self.solar_constant * cos_zenith * transmission
        
        # Calculate surface orientation factor
        sun_azimuth = torch.rad2deg(hour_angle)
        fsurt = self.calculate_surface_orientation_factor(cos_zenith, slope, sun_azimuth, panel_azimuth)
        
        # Calculate cell temperature
        cell_temp = self.calculate_cell_temperature(ambient_temp, irradiance)
        
        # Calculate temperature correction factor
        temp_correction = 1 + self.temp_coefficient * (cell_temp - self.ref_temperature)
        
        # Calculate total efficiency with temperature correction
        # η = ηref * fsurt * [1 + β * (Tc - Tref)]
        total_efficiency = ref_efficiency * fsurt * temp_correction
        
        # Apply daylight mask
        sunrise, sunset, is_polar_day, is_polar_night = self.calculate_sunrise_sunset(latitude, day_of_year)
        is_daytime = is_polar_day | ((hour_of_day >= sunrise) & (hour_of_day <= sunset))
        total_efficiency = torch.where(is_daytime, total_efficiency, torch.zeros_like(total_efficiency))
        
        return torch.clamp(total_efficiency, min=0.0)

    def calculate_irradiance(self, latitude, time, slope=0, panel_azimuth=0):
        """Calculate the total solar irradiance at the surface under ideal conditions."""
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        sunrise, sunset, is_polar_day, is_polar_night = self.calculate_sunrise_sunset(latitude, day_of_year)
        is_daytime = is_polar_day | ((hour_of_day >= sunrise) & (hour_of_day <= sunset))
        
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)
        
        air_mass = self.calculate_air_mass(cos_zenith)
        transmission = self.calculate_atmospheric_transmission(air_mass)
        
        surface_orientation = self.calculate_surface_orientation_factor(
            cos_zenith, slope, hour_angle, panel_azimuth
        )
        
        irradiance = self.solar_constant * transmission * surface_orientation
        irradiance = torch.where(is_daytime, irradiance, torch.zeros_like(irradiance))
        
        return torch.clamp(irradiance, min=0.0)

def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics between true and predicted values."""
    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item()
    }