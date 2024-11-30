import numpy as np
import torch

class DataProcessor:
    def __init__(self):
        self.input_scaler = None
        self.output_scaler = None
        
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
        return (data - self.input_scaler['mean']) / self.input_scaler['std']
    
    def denormalize_data(self, data):
        """Denormalize predictions"""
        return data * self.input_scaler['std'] + self.input_scaler['mean']
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data"""
        lat = np.random.uniform(-90, 90, n_samples)
        lon = np.random.uniform(-180, 180, n_samples)
        time = np.random.uniform(0, 24, n_samples)
        slope = np.random.uniform(0, 45, n_samples)
        aspect = np.random.uniform(0, 360, n_samples)
        atm = np.random.uniform(0.5, 1.0, n_samples)
        
        return self.prepare_data(lat, lon, time, slope, aspect, atm)
