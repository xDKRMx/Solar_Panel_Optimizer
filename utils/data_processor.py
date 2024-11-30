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
        """Convert predictions to physical units"""
        # Predictions are already in correct efficiency range from model
        if scale_irradiance:
            return predictions * self.irradiance_scale
        return predictions
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        lat = np.random.uniform(-90, 90, n_samples)
        lon = np.random.uniform(-180, 180, n_samples)
        time = np.random.uniform(0, 24, n_samples)
        slope = np.random.uniform(0, 45, n_samples)
        aspect = np.random.uniform(0, 360, n_samples)
        atm = np.random.uniform(0.5, 1.0, n_samples)
        
        return self.prepare_data(lat, lon, time, slope, aspect, atm)
