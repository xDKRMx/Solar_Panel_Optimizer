import numpy as np

class SolarIrradianceCalculator:
    def __init__(self):
        self.solar_constant = 1367  # W/m²

    def solar_declination(self, day_number):
        """Calculate solar declination angle"""
        return 23.45 * np.sin(2 * np.pi * (284 + day_number) / 365)

    def hour_angle(self, hour, longitude):
        """Calculate hour angle"""
        return 15 * (hour - 12) + longitude

    def cos_incidence_angle(self, latitude, declination, hour_angle, slope, aspect):
        """Calculate cosine of solar incidence angle"""
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        slope_rad = np.radians(slope)
        aspect_rad = np.radians(aspect)
        
        return (np.sin(lat_rad) * np.sin(decl_rad) * np.cos(slope_rad) -
                np.cos(lat_rad) * np.sin(decl_rad) * np.sin(slope_rad) * np.cos(aspect_rad) +
                np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad) * np.cos(slope_rad) +
                np.sin(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad) * np.sin(slope_rad) * np.cos(aspect_rad) +
                np.cos(decl_rad) * np.sin(hour_rad) * np.sin(slope_rad) * np.sin(aspect_rad))

    def calculate_irradiance(self, latitude, longitude, day_number, hour, slope, aspect, atm_transmission):
        """Calculate solar irradiance"""
        declination = self.solar_declination(day_number)
        hour_ang = self.hour_angle(hour, longitude)
        cos_inc = self.cos_incidence_angle(latitude, declination, hour_ang, slope, aspect)
        
        # Direct normal irradiance
        dni = self.solar_constant * atm_transmission
        
        # Surface irradiance
        irradiance = dni * max(0, cos_inc)
        
        return irradiance
