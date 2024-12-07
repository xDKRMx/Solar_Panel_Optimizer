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
        self.base_noct = 45.0  # Base Nominal Operating Cell Temperature, °C
        
        # Seasonal adjustment parameters
        self.seasonal_temp_variation = 15.0  # Maximum temperature variation across seasons (°C)
        self.seasonal_noct_variation = 3.0  # NOCT variation across seasons (°C)
        self.seasonal_extinction_variation = 0.02  # Seasonal variation in extinction coefficient

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

    def calculate_atmospheric_transmission(self, air_mass, day_of_year, latitude):
        """Calculate the atmospheric transmission factor (T) with seasonal variations.
        
        Args:
            air_mass: Calculated air mass
            day_of_year: Day of the year (1-365)
            latitude: Location latitude in degrees
            
        Returns:
            Atmospheric transmission factor
        """
        # Calculate seasonal extinction coefficient
        abs_lat = abs(float(latitude))
        phase_shift = 0 if latitude >= 0 else 182.5
        season_factor = torch.cos(2 * torch.pi * ((day_of_year + phase_shift) / 365))
        lat_scale = (abs_lat / 90) ** 0.5
        
        # Adjust extinction coefficient for seasonal variations
        seasonal_extinction = (self.extinction_coefficient + 
                             self.seasonal_extinction_variation * season_factor * lat_scale)
        
        # Calculate transmission components with seasonal adjustment
        base_transmission = torch.exp(-seasonal_extinction * air_mass)
        rayleigh = torch.exp(-0.0903 * air_mass ** 0.84)
        aerosol = torch.exp(-0.08 * (1 + 0.1 * season_factor) * air_mass ** 0.95)
        ozone = torch.exp(-0.0042 * (1 + 0.05 * season_factor) * air_mass ** 0.95)
        
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

    def calculate_seasonal_ambient_temp(self, base_temp, day_of_year, latitude):
        """Calculate seasonally adjusted ambient temperature.
        
        Args:
            base_temp: Base ambient temperature in °C
            day_of_year: Day of the year (1-365)
            latitude: Location latitude in degrees
            
        Returns:
            Seasonally adjusted ambient temperature in °C
        """
        # Convert latitude to absolute value for hemisphere adjustment
        abs_lat = abs(float(latitude))
        
        # Adjust phase based on hemisphere (inverted seasons in Southern hemisphere)
        phase_shift = 0 if latitude >= 0 else 182.5
        
        # Calculate seasonal temperature variation
        # Maximum variation occurs on day 172 (summer) and day 355 (winter)
        season_factor = torch.cos(2 * torch.pi * ((day_of_year + phase_shift) / 365))
        
        # Latitude scaling (stronger seasonal effects at higher latitudes)
        lat_scale = (abs_lat / 90) ** 0.5
        
        # Calculate temperature adjustment
        temp_adjustment = self.seasonal_temp_variation * season_factor * lat_scale
        
        return base_temp + temp_adjustment

    def calculate_seasonal_noct(self, day_of_year, latitude):
        """Calculate seasonally adjusted NOCT.
        
        Args:
            day_of_year: Day of the year (1-365)
            latitude: Location latitude in degrees
            
        Returns:
            Seasonally adjusted NOCT in °C
        """
        # Similar seasonal variation as ambient temperature
        abs_lat = abs(float(latitude))
        phase_shift = 0 if latitude >= 0 else 182.5
        
        # NOCT varies with season due to average ambient conditions
        season_factor = torch.cos(2 * torch.pi * ((day_of_year + phase_shift) / 365))
        lat_scale = (abs_lat / 90) ** 0.5
        
        noct_adjustment = self.seasonal_noct_variation * season_factor * lat_scale
        return self.base_noct + noct_adjustment

    def calculate_cell_temperature(self, ambient_temp, irradiance, day_of_year, latitude):
        """Calculate cell temperature using enhanced NOCT method with seasonal variations.
        
        Args:
            ambient_temp: Base ambient temperature in °C
            irradiance: Solar irradiance in W/m²
            day_of_year: Day of the year (1-365)
            latitude: Location latitude in degrees
            
        Returns:
            Cell temperature in °C
        """
        # Get seasonally adjusted ambient temperature and NOCT
        adj_ambient_temp = self.calculate_seasonal_ambient_temp(ambient_temp, day_of_year, latitude)
        adj_noct = self.calculate_seasonal_noct(day_of_year, latitude)
        
        # Enhanced temperature calculation with improved thermal model
        # Standard Test Conditions (STC) irradiance is 1000 W/m²
        irradiance_factor = irradiance / 1000
        
        # Wind speed effect (simplified model)
        # Assume higher wind speeds in winter, lower in summer
        season_factor = torch.cos(2 * torch.pi * (day_of_year / 365))
        wind_factor = 1.0 + 0.2 * season_factor  # ±20% variation
        
        # Calculate cell temperature using improved NOCT method
        # Temperature rise = (NOCT - 20°C) * (Irradiance/800) * (1 - η_thermal) * wind_factor
        # η_thermal represents thermal losses (temperature dependent)
        base_thermal_efficiency = 0.9
        thermal_efficiency = base_thermal_efficiency - 0.002 * (adj_ambient_temp - 25)  # Temperature dependency
        thermal_efficiency = torch.clamp(thermal_efficiency, min=0.85, max=0.95)
        
        temperature_rise = (adj_noct - 20) * (irradiance / 800) * (1 - thermal_efficiency) / wind_factor
        
        return adj_ambient_temp + temperature_rise

    def calculate_efficiency(self, latitude, time, slope=0, panel_azimuth=0, ref_efficiency=0.15, ambient_temp=25):
        """Calculate temperature-corrected solar panel efficiency with enhanced seasonal effects.
        
        Args:
            latitude: Location latitude in degrees
            time: Time of day (hour)
            slope: Panel slope angle in degrees (β)
            panel_azimuth: Panel azimuth angle in degrees (φp)
            ref_efficiency: Reference efficiency under STC (default 0.15 or 15%)
            ambient_temp: Base ambient temperature in °C (default 25°C)
            
        Returns:
            Temperature-corrected total efficiency including seasonal variations
        """
        # Convert inputs to tensors
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        ambient_temp = torch.as_tensor(ambient_temp, dtype=torch.float32)
        
        # Calculate day of year and time components with proper range handling
        day_of_year = torch.floor(time / 24 * 365)
        day_of_year = torch.clamp(day_of_year, min=1, max=365)
        hour_of_day = time % 24
        
        # Calculate base irradiance for cell temperature calculation
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)
        
        # Calculate solar position with seasonal variations
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        
        # Calculate zenith angle
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)
        
        # Calculate irradiance with seasonal effects
        air_mass = self.calculate_air_mass(cos_zenith)
        transmission = self.calculate_atmospheric_transmission(air_mass, day_of_year, latitude)
        
        # Apply orbital variation to solar constant
        orbit_factor = 1 + 0.033 * torch.cos(2 * torch.pi * day_of_year / 365)
        seasonal_solar_constant = self.solar_constant * orbit_factor
        
        irradiance = seasonal_solar_constant * cos_zenith * transmission
        
        # Calculate surface orientation factor
        sun_azimuth = torch.rad2deg(hour_angle)
        fsurt = self.calculate_surface_orientation_factor(cos_zenith, slope, sun_azimuth, panel_azimuth)
        
        # Calculate cell temperature with seasonal variations
        cell_temp = self.calculate_cell_temperature(ambient_temp, irradiance, day_of_year, latitude)
        
        # Calculate temperature coefficient with seasonal adjustment
        # Temperature coefficient becomes more negative at higher temperatures
        temp_diff = cell_temp - self.ref_temperature
        adjusted_temp_coef = self.temp_coefficient * (1 - 0.0005 * temp_diff)  # Slight adjustment for temperature dependence
        
        # Calculate temperature correction factor with seasonal adjustment
        temp_correction = 1 + adjusted_temp_coef * temp_diff
        
        # Calculate seasonal efficiency adjustment
        # Efficiency typically slightly better in cooler seasons
        abs_lat = abs(float(latitude))
        phase_shift = 0 if latitude >= 0 else 182.5
        season_factor = torch.cos(2 * torch.pi * ((day_of_year + phase_shift) / 365))
        lat_scale = (abs_lat / 90) ** 0.5
        seasonal_efficiency_adj = 1 + 0.01 * season_factor * lat_scale  # ±1% seasonal variation
        
        # Calculate total efficiency with enhanced temperature correction
        # η = ηref * fsurt * [1 + β * (Tc - Tref)]
        # Temperature coefficient becomes more negative at higher temperatures
        temp_diff = cell_temp - self.ref_temperature
        # Enhanced temperature coefficient model that varies with temperature
        beta_temp = self.temp_coefficient * (1 + 0.002 * abs(temp_diff))  # More negative at higher temperature differences
        temp_correction = 1 + beta_temp * temp_diff
        
        # Calculate total efficiency with physics-based corrections
        total_efficiency = ref_efficiency * fsurt * temp_correction
        
        # Apply daylight mask with seasonal sunrise/sunset times
        sunrise, sunset, is_polar_day, is_polar_night = self.calculate_sunrise_sunset(latitude, day_of_year)
        is_daytime = is_polar_day | ((hour_of_day >= sunrise) & (hour_of_day <= sunset))
        total_efficiency = torch.where(is_daytime, total_efficiency, torch.zeros_like(total_efficiency))
        
        return torch.clamp(total_efficiency, min=0.0)

    def calculate_irradiance(self, latitude, time, slope=0, panel_azimuth=0):
        """Calculate the total solar irradiance at the surface under ideal conditions with seasonal variations."""
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        # Calculate solar position with seasonal effects
        sunrise, sunset, is_polar_day, is_polar_night = self.calculate_sunrise_sunset(latitude, day_of_year)
        is_daytime = is_polar_day | ((hour_of_day >= sunrise) & (hour_of_day <= sunset))
        
        declination = self.calculate_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(hour_of_day)
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)
        
        # Enhanced air mass calculation with seasonal adjustments
        air_mass = self.calculate_air_mass(cos_zenith)
        
        # Calculate atmospheric transmission with seasonal variations
        transmission = self.calculate_atmospheric_transmission(air_mass, day_of_year, latitude)
        
        # Calculate surface orientation with seasonal optimization
        surface_orientation = self.calculate_surface_orientation_factor(
            cos_zenith, slope, hour_angle, panel_azimuth
        )
        
        # Apply seasonal variations to final irradiance calculation
        # Earth's orbit eccentricity effect on solar constant
        orbit_factor = 1 + 0.033 * torch.cos(2 * torch.pi * day_of_year / 365)
        seasonal_solar_constant = self.solar_constant * orbit_factor
        
        irradiance = seasonal_solar_constant * transmission * surface_orientation
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