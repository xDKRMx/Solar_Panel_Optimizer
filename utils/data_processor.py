import numpy as np
import torch

class DataProcessor:
    def __init__(self):
        # Physical constants for non-dimensionalization
        self.earth_radius = 6371.0  # km
        self.day_period = 24.0  # hours
        self.solar_constant = 1367.0  # W/m²
        
        # Initialize scalers
        self.length_scale = self.earth_radius
        self.time_scale = self.day_period
        self.irradiance_scale = self.solar_constant
        
    def prepare_data(self, lat, lon, time, slope, aspect, atm, cloud_cover=None, wavelength=None):
        """Prepare and non-dimensionalize data for PINN model"""
        # Convert to numpy arrays if not already
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        time = np.asarray(time)
        slope = np.asarray(slope)
        aspect = np.asarray(aspect)
        atm = np.asarray(atm)
        
        # Set default values if not provided
        if cloud_cover is None:
            cloud_cover = np.zeros_like(lat)
        else:
            cloud_cover = np.asarray(cloud_cover)
        
        if wavelength is None:
            wavelength = np.full_like(lat, 0.5)  # Default wavelength in μm
        else:
            wavelength = np.asarray(wavelength)
        
        # Non-dimensionalize the data
        time_nd = time / self.time_scale
        wavelength_nd = wavelength / 0.5  # Normalize to reference wavelength
        
        # Angles (lat, lon, slope, aspect) are already non-dimensional
        # Atmospheric transmission, cloud cover are already non-dimensional
        
        data = np.column_stack([
            lat, lon, time_nd, slope, aspect, atm, 
            cloud_cover, wavelength_nd
        ])
        return torch.FloatTensor(data)
    
    def normalize_data(self, data):
        """Normalize input data to [-1, 1] range"""
        # Define normalization ranges for each variable
        ranges = np.array([
            [-90, 90],      # latitude
            [-180, 180],    # longitude
            [0, 1],        # normalized time
            [0, 90],       # slope
            [0, 360],      # aspect
            [0, 1]         # atmospheric transmission
        ])
        
        # Normalize to [-1, 1]
        normalized = 2 * (data - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1
        return normalized
    
    def denormalize_predictions(self, predictions, scale_irradiance=True):
        """Convert normalized predictions back to physical units"""
        # First denormalize from [-1, 1] to [0, 1]
        predictions_01 = (predictions + 1) / 2
        
        # Then scale to physical units
        if scale_irradiance:
            return predictions_01 * self.irradiance_scale
        return predictions_01
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data with enhanced nighttime sampling"""
        # Regular daytime samples
        day_samples = int(n_samples * 0.7)  # 70% daytime conditions
        lat_day = np.random.uniform(-90, 90, day_samples)
        lon_day = np.random.uniform(-180, 180, day_samples)
        time_day = np.random.uniform(6, 18, day_samples)  # Daytime hours
        slope_day = np.random.uniform(0, 45, day_samples)
        aspect_day = np.random.uniform(0, 360, day_samples)
        atm_day = np.random.uniform(0.5, 1.0, day_samples)

        # Nighttime samples
        night_samples = n_samples - day_samples  # 30% nighttime conditions
        lat_night = np.random.uniform(-90, 90, night_samples)
        lon_night = np.random.uniform(-180, 180, night_samples)
        time_night = np.random.uniform(18, 30, night_samples)  # Evening to morning
        time_night = np.where(time_night >= 24, time_night - 24, time_night)  # Wrap to [0, 24]
        slope_night = np.random.uniform(0, 45, night_samples)
        aspect_night = np.random.uniform(0, 360, night_samples)
        atm_night = np.random.uniform(0.5, 1.0, night_samples)

        # Combine day and night samples
        lat = np.concatenate([lat_day, lat_night])
        lon = np.concatenate([lon_day, lon_night])
        time = np.concatenate([time_day, time_night])
        slope = np.concatenate([slope_day, slope_night])
        aspect = np.concatenate([aspect_day, aspect_night])
        atm = np.concatenate([atm_day, atm_night])
        
        return self.prepare_data(lat, lon, time, slope, aspect, atm)
