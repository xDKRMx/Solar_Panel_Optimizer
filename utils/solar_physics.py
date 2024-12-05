import numpy as np

class SolarIrradianceCalculator:
    def __init__(self):
        # Physical constants
        self.solar_constant = 1367.0  # W/m²
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.planck_constant = 6.626e-34  # J·s
        self.speed_of_light = 2.998e8  # m/s
        self.boltzmann_constant = 1.381e-23  # J/K
        self.atmospheric_extinction = 0.1  # Atmospheric extinction coefficient
        self.earth_radius = 6371000  # Earth radius in meters
        self.air_mass_coefficient = 1.0  # Air mass coefficient
        
        # Model parameters
        self.beta = 0.1  # Default aerosol optical thickness
        self.alpha = 1.3  # Default Ångström exponent
        self.cloud_alpha = 0.75  # Empirical cloud transmission parameter
        self.albedo = 0.2  # Ground reflectance

    def spectral_irradiance(self, wavelength, temperature):
        '''Planck's law for spectral radiance'''
        c1 = 2 * self.planck_constant * self.speed_of_light**2
        c2 = self.planck_constant * self.speed_of_light / self.boltzmann_constant
        return c1 / (wavelength**5 * (np.exp(c2/(wavelength*temperature)) - 1))

    def calculate_hour_angle(self, longitude, time):
        """Calculate hour angle based on longitude and time"""
        # Convert time to hour angle (15 degrees per hour from solar noon)
        solar_noon = 12.0  # Local solar noon
        hour_angle = 15.0 * (time - solar_noon)  # 15 degrees per hour
        
        # Adjust for longitude
        # Each degree of longitude equals 4 minutes of time
        time_offset = longitude * 4.0 / 60.0  # Convert to hours
        hour_angle += time_offset * 15.0  # Convert time offset to degrees
        
        return np.radians(hour_angle)

    def calculate_solar_zenith(self, latitude, longitude, time):
        """Calculate solar zenith angle"""
        hour_angle = self.calculate_hour_angle(longitude, time)
        declination = self.calculate_declination(time)
        
        lat_rad = np.radians(latitude)
        hour_rad = np.radians(hour_angle)
        decl_rad = np.radians(declination)
        
        cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) +
                     np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        return np.arccos(np.clip(cos_zenith, -1.0, 1.0))

    def calculate_solar_azimuth(self, latitude, longitude, time):
        """Calculate solar azimuth angle"""
        hour_angle = self.calculate_hour_angle(longitude, time)
        declination = self.calculate_declination(time)
        
        lat_rad = np.radians(latitude)
        hour_rad = np.radians(hour_angle)
        decl_rad = np.radians(declination)
        
        cos_phi = ((np.sin(decl_rad) * np.cos(lat_rad) -
                   np.cos(decl_rad) * np.sin(lat_rad) * np.cos(hour_rad)) /
                  np.sin(self.calculate_solar_zenith(latitude, longitude, time)))
        return np.arccos(np.clip(cos_phi, -1.0, 1.0))

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass using improved Kasten and Young formula"""
        z = np.degrees(zenith_angle)
        return 1 / (np.cos(zenith_angle) + 0.50572 * (96.07995 - z)**(-1.6364))

    def calculate_rayleigh_scattering(self, air_mass, pressure):
        """Calculate Rayleigh scattering transmittance"""
        return np.exp(-0.1128 * air_mass * (pressure / 101325.0))

    def calculate_water_vapor_absorption(self, humidity, air_mass):
        """Calculate water vapor absorption"""
        w = humidity * 4.12  # Precipitable water vapor content
        return np.exp(-0.2385 * w * air_mass / (1 + 20.07 * w * air_mass)**0.45)

    def calculate_ozone_absorption(self, air_mass):
        """Calculate ozone absorption"""
        L = 0.35  # Typical ozone column thickness (cm)
        return np.exp(-0.0365 * L * air_mass)

    def calculate_incidence_angle(self, zenith_angle, azimuth, slope, aspect):
        """Calculate incidence angle on tilted surface"""
        cos_zenith = np.cos(zenith_angle)
        sin_zenith = np.sin(zenith_angle)
        cos_slope = np.cos(np.radians(slope))
        sin_slope = np.sin(np.radians(slope))
        cos_azimuth_diff = np.cos(azimuth - np.radians(aspect))
        
        return np.arccos(cos_zenith * cos_slope + 
                        sin_zenith * sin_slope * cos_azimuth_diff)

    def calculate_diffuse_component(self, direct_normal, zenith_angle):
        """Calculate diffuse radiation component"""
        sky_factor = 0.5 * (1 + np.cos(zenith_angle))  # Sky view factor
        return 0.2 * direct_normal * sky_factor  # Simplified diffuse model

    def combine_components(self, direct_normal, diffuse_horizontal, 
                         incidence_angle, slope):
        """Combine radiation components"""
        # Direct component on tilted surface
        direct_tilted = direct_normal * np.cos(incidence_angle)
        
        # Diffuse component with tilt factor
        tilt_factor = (1 + np.cos(np.radians(slope))) / 2
        diffuse_tilted = diffuse_horizontal * tilt_factor
        
        # Ground-reflected component
        ground_reflection = direct_normal * self.albedo * (1 - tilt_factor)
        
        return direct_tilted + diffuse_tilted + ground_reflection

    def calculate_declination(self, time):
        """Calculate solar declination angle"""
        # Approximate solar declination using day number
        day_number = time  # Assuming time is day number (1-365)
        return 23.45 * np.sin(2 * np.pi * (284 + day_number) / 365)

    def calculate_irradiance(self, latitude, longitude, time, slope, aspect,
                           temperature=288.15, pressure=101325.0, humidity=0.5):
        """Complete solar irradiance calculation"""
        # Solar position
        zenith_angle = self.calculate_solar_zenith(latitude, longitude, time)
        azimuth = self.calculate_solar_azimuth(latitude, longitude, time)
        
        # Atmospheric transmission
        air_mass = self.calculate_air_mass(zenith_angle)
        rayleigh_scattering = self.calculate_rayleigh_scattering(air_mass, pressure)
        water_vapor_absorption = self.calculate_water_vapor_absorption(humidity, air_mass)
        ozone_absorption = self.calculate_ozone_absorption(air_mass)
        
        # Direct and diffuse components
        direct_normal = self.solar_constant * np.exp(-air_mass * 
            (rayleigh_scattering * water_vapor_absorption * ozone_absorption))
        diffuse_horizontal = self.calculate_diffuse_component(direct_normal, zenith_angle)
        
        # Surface orientation effects
        incidence_angle = self.calculate_incidence_angle(zenith_angle, azimuth, 
                                                       slope, aspect)
        
        return self.combine_components(direct_normal, diffuse_horizontal,
                                     incidence_angle, slope)
