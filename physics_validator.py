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

    def calculate_air_mass(self, zenith_angle, day_of_year, latitude):
        """Calculate air mass (M) with DOY effects and enhanced seasonal corrections."""
        zenith_angle_deg = torch.rad2deg(torch.acos(zenith_angle))
        
        # Basic air mass calculation (Kasten-Young formula)
        basic_am = 1 / (torch.cos(torch.deg2rad(zenith_angle_deg)) +
                       0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
        
        # Seasonal pressure correction
        # Atmospheric pressure varies with season, affecting air mass
        pressure_correction = 1.0 + 0.02 * torch.cos(2 * torch.pi * (day_of_year - 1) / 365)
        
        # Latitude-based seasonal correction
        # Higher latitudes experience more significant seasonal variations
        lat_rad = torch.deg2rad(latitude)
        seasonal_lat_factor = 1.0 + 0.1 * torch.abs(torch.sin(lat_rad)) * \
                            torch.cos(2 * torch.pi * (day_of_year - 1) / 365)
        
        # High latitude correction with seasonal component
        high_lat_correction = torch.where(
            zenith_angle_deg > 60,
            1 + (zenith_angle_deg - 60) * 0.15 * seasonal_lat_factor / 30,
            torch.ones_like(zenith_angle_deg)
        )
        
        # Combined air mass calculation
        air_mass = basic_am * pressure_correction / high_lat_correction
        
        return torch.clamp(air_mass, min=1.0)

    def calculate_atmospheric_transmission(self, air_mass, day_of_year):
        """Calculate the atmospheric transmission factor (T) with seasonal variations."""
        # Calculate seasonal variation factor (peaks at summer solstice)
        seasonal_factor = 1.0 + 0.03 * torch.cos(2 * torch.pi * (day_of_year - 172) / 365)
        
        # Base atmospheric transmission with seasonal adjustment
        base_transmission = torch.exp(-self.extinction_coefficient * air_mass * seasonal_factor)
        
        # Seasonal adjustments for different atmospheric components
        summer_solstice = 172  # June 21st
        winter_solstice = 355  # December 21st
        
        # Calculate seasonal weights
        summer_weight = 0.5 + 0.5 * torch.cos(2 * torch.pi * (day_of_year - summer_solstice) / 365)
        winter_weight = 1.0 - summer_weight
        
        # Seasonal Rayleigh scattering (stronger in winter due to denser atmosphere)
        rayleigh_winter = torch.exp(-0.1083 * air_mass ** 0.84)  # Enhanced winter scattering
        rayleigh_summer = torch.exp(-0.0903 * air_mass ** 0.84)  # Standard summer scattering
        rayleigh = rayleigh_summer * summer_weight + rayleigh_winter * winter_weight
        
        # Seasonal aerosol effects (stronger in summer)
        aerosol_summer = torch.exp(-0.1 * air_mass ** 0.95)  # Enhanced summer aerosols
        aerosol_winter = torch.exp(-0.06 * air_mass ** 0.95)  # Reduced winter aerosols
        aerosol = aerosol_summer * summer_weight + aerosol_winter * winter_weight
        
        # Ozone variation (stronger in spring/summer)
        ozone = torch.exp(-0.0042 * air_mass ** 0.95 * seasonal_factor)
        
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
        
        # Calculate irradiance with enhanced DOY effects
        air_mass = self.calculate_air_mass(cos_zenith, day_of_year, latitude)
        transmission = self.calculate_atmospheric_transmission(air_mass, day_of_year)
        
        # Enhanced seasonal solar constant variation
        # Earth's elliptical orbit causes ~3.3% annual variation in solar radiation
        orbital_factor = 1 + 0.033 * torch.cos(2 * torch.pi * (day_of_year - 2) / 365)
        irradiance = self.solar_constant * orbital_factor * cos_zenith * transmission
        
        # Calculate surface orientation factor
        sun_azimuth = torch.rad2deg(hour_angle)
        fsurt = self.calculate_surface_orientation_factor(cos_zenith, slope, sun_azimuth, panel_azimuth)
        
        # Calculate cell temperature with seasonal adjustments
        cell_temp = self.calculate_cell_temperature(ambient_temp, irradiance)
        
        # Seasonal adjustment to temperature coefficient
        # Temperature coefficient typically becomes more negative at higher temperatures
        seasonal_temp_factor = 1.0 + 0.1 * torch.cos(2 * torch.pi * (day_of_year - 172) / 365)
        adjusted_temp_coeff = self.temp_coefficient * seasonal_temp_factor
        
        # Calculate temperature correction factor with seasonal adjustment
        temp_correction = 1 + adjusted_temp_coeff * (cell_temp - self.ref_temperature)
        
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