import numpy as np
import torch
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.input_scaler = None
        self.output_scaler = None
        
        # Real solar installation site ranges (focusing on high-potential regions)
        self.site_ranges = {
            'latitude': (-35, 35),  # Major solar belt
            'longitude': (-180, 180)
        }
        
    def get_optimal_tilt(self, latitude):
        """Calculate optimal tilt angle based on latitude"""
        # General rule: latitude * 0.76 + 3.1 degrees
        return abs(latitude) * 0.76 + 3.1
    
    def get_optimal_aspect(self, latitude):
        """Calculate optimal aspect (azimuth) based on hemisphere"""
        # Northern hemisphere: 180° (south-facing)
        # Southern hemisphere: 0° (north-facing)
        return 180 if latitude > 0 else 0
    
    def get_daylight_hours(self, latitude, day_number):
        """Calculate daylight hours based on latitude and day of year"""
        declination = 23.45 * np.sin(np.radians(360/365 * (day_number - 81)))
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        
        cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
        
        daylight_hours = 2 * np.degrees(np.arccos(cos_hour_angle)) / 15
        return daylight_hours
    
    def prepare_data(self, lat, lon, time, slope, aspect, atm):
        """Prepare data for PINN model"""
        data = np.column_stack([lat, lon, time, slope, aspect, atm])
        return torch.FloatTensor(data)
    
    def normalize_data(self, data):
        """Normalize input data"""
        if self.input_scaler is None:
            self.input_scaler = {
                'mean': data.mean(axis=0),
                'std': data.std(axis=0)
            }
        return (data - self.input_scaler['mean']) / (self.input_scaler['std'] + 1e-8)
    
    def denormalize_data(self, data):
        """Denormalize predictions"""
        return data * self.input_scaler['std'] + self.input_scaler['mean']
    
    def generate_training_data(self, n_samples=1000):
        """Generate realistic synthetic training data"""
        # Generate latitudes within solar installation regions
        lat = np.random.uniform(
            self.site_ranges['latitude'][0],
            self.site_ranges['latitude'][1],
            n_samples
        )
        
        # Generate corresponding longitudes
        lon = np.random.uniform(
            self.site_ranges['longitude'][0],
            self.site_ranges['longitude'][1],
            n_samples
        )
        
        # Generate realistic day numbers and times
        day_numbers = np.random.randint(1, 366, n_samples)
        
        # Calculate daylight hours for each latitude/day combination
        daylight_hours = np.array([self.get_daylight_hours(lat_i, day_i) 
                                 for lat_i, day_i in zip(lat, day_numbers)])
        
        # Generate times within daylight hours
        time = np.random.uniform(
            12 - daylight_hours/2,
            12 + daylight_hours/2,
            n_samples
        )
        
        # Generate slopes based on optimal values with some variation
        optimal_slopes = np.array([self.get_optimal_tilt(lat_i) for lat_i in lat])
        slope = np.random.normal(optimal_slopes, 5.0)  # 5° standard deviation
        slope = np.clip(slope, 0, 90)  # Ensure physical bounds
        
        # Generate aspects based on hemisphere with some variation
        optimal_aspects = np.array([self.get_optimal_aspect(lat_i) for lat_i in lat])
        aspect = np.random.normal(optimal_aspects, 20.0)  # 20° standard deviation
        aspect = np.mod(aspect, 360)  # Ensure 0-360° range
        
        # Generate realistic atmospheric transmission values
        atm = np.random.uniform(0.6, 0.8, n_samples)  # Typical range for clear sky
        
        return self.prepare_data(lat, lon, time, slope, aspect, atm)
