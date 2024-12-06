import math
import datetime

class SolarAstronomyCalculator:
    """
    A utility class for calculating solar astronomy-related parameters including
    sunrise, sunset times, and solar declination angle.
    """
    
    def __init__(self):
        self.DEG_TO_RAD = math.pi / 180
        self.RAD_TO_DEG = 180 / math.pi
        
    def calculate_solar_declination(self, day_of_year):
        """
        Calculate the solar declination angle (in radians) for a given day of the year.
        
        Args:
            day_of_year (int): Day of the year (1-365)
            
        Returns:
            float: Solar declination angle in radians
        """
        return 23.45 * self.DEG_TO_RAD * math.sin(
            (2 * math.pi / 365) * (day_of_year - 81)
        )
    
    def calculate_hour_angle(self, latitude, solar_declination):
        """
        Calculate the hour angle at sunrise/sunset.
        
        Args:
            latitude (float): Latitude in degrees
            solar_declination (float): Solar declination in radians
            
        Returns:
            float: Hour angle in radians
        """
        lat_rad = latitude * self.DEG_TO_RAD
        cos_h = -math.tan(lat_rad) * math.tan(solar_declination)
        
        # Ensure cos_h is within valid range [-1, 1]
        if cos_h < -1:
            cos_h = -1  # Polar day
        elif cos_h > 1:
            cos_h = 1   # Polar night
            
        return math.acos(cos_h)
    
    def equation_of_time(self, day_of_year):
        """
        Calculate the equation of time correction in hours.
        
        Args:
            day_of_year (int): Day of the year (1-365)
            
        Returns:
            float: Time correction in hours
        """
        B = 2 * math.pi * (day_of_year - 81) / 365
        return (9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)) / 60
        
    def calculate_sunrise_sunset(self, latitude, day_of_year, timezone_offset=0, longitude=-122.4194):
        """
        Calculate sunrise and sunset times for a given latitude and day of year.
        
        Args:
            latitude (float): Latitude in degrees
            day_of_year (int): Day of the year (1-365)
            timezone_offset (float): Timezone offset from UTC in hours
            longitude (float): Longitude in degrees (negative for western hemisphere)
            
        Returns:
            tuple: (sunrise_time, sunset_time) in hours (local time)
        """
        solar_declination = self.calculate_solar_declination(day_of_year)
        hour_angle = self.calculate_hour_angle(latitude, solar_declination)
        
        # Convert hour angle to hours
        hours_from_noon = hour_angle * self.RAD_TO_DEG / 15
        
        # Calculate equation of time correction
        eot = self.equation_of_time(day_of_year)
        
        # Calculate longitude correction (4 minutes per degree, converted to hours)
        longitude_correction = -longitude / 15
        
        # Calculate solar noon including all corrections
        solar_noon = 12 - eot + longitude_correction
        
        # Calculate sunrise and sunset times
        sunrise = (solar_noon - hours_from_noon + timezone_offset) % 24
        sunset = (solar_noon + hours_from_noon + timezone_offset) % 24
        
        return sunrise, sunset
    
    def get_day_of_year(self, date=None):
        """
        Get the day of year (1-365) for a given date or current date.
        
        Args:
            date (datetime.date, optional): Date to calculate day of year for
            
        Returns:
            int: Day of year (1-365)
        """
        if date is None:
            date = datetime.date.today()
        return date.timetuple().tm_yday
    
    def format_time(self, hours):
        """
        Format decimal hours as HH:MM time string.
        
        Args:
            hours (float): Time in decimal hours
            
        Returns:
            str: Formatted time string (HH:MM)
        """
        # Handle hours outside 0-24 range
        while hours < 0:
            hours += 24
        while hours >= 24:
            hours -= 24
            
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h:02d}:{m:02d}"
