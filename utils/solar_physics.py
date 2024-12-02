import numpy as np

class SolarIrradianceCalculator:
    def __init__(self):
        self.solar_constant = 1367  # W/m²
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        self.cloud_alpha = 0.75  # Empirical cloud transmission parameter
        self.ref_wavelength = 0.5  # μm, reference wavelength

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

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using Kasten and Young's formula"""
        zenith_deg = np.degrees(zenith_angle)
        return 1.0 / (np.cos(zenith_angle) + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))

    def calculate_optical_depth(self, wavelength):
        """Calculate wavelength-dependent optical depth using Ångström turbidity formula"""
        return self.beta * (wavelength / self.ref_wavelength) ** (-self.alpha)

    def calculate_irradiance(self, latitude, longitude, day_number, hour, slope, aspect, atm_transmission, cloud_cover=0.0, wavelength=0.5):
        """Calculate solar irradiance with advanced atmospheric effects"""
        
        declination = self.solar_declination(day_number)
        hour_ang = self.hour_angle(hour, longitude)
        cos_inc = self.cos_incidence_angle(latitude, declination, hour_ang, slope, aspect)
        
        
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_ang)
        cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) +
                     np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        
        
        if cos_zenith <= 0:
            return 0.0
            
        
        cos_zenith = np.clip(cos_zenith, 0.001, 1.0)
        zenith_angle = np.arccos(cos_zenith)
        
       
        air_mass = self.calculate_air_mass(zenith_angle)
        optical_depth = self.calculate_optical_depth(wavelength)
        
       
        cloud_transmission = 1.0 - self.cloud_alpha * (cloud_cover ** 3)
        
        
        dni = self.solar_constant * np.exp(-optical_depth * air_mass) * atm_transmission * cloud_transmission
        
       
        irradiance = dni * np.maximum(0, cos_inc)
        
        return irradiance
