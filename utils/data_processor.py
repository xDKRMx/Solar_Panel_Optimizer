import numpy as np
import torch


class DataProcessor:

    def __init__(self):
        # Physical constants for non-dimensionalization
        self.earth_radius = 6371.0  # km
        self.day_period = 24.0  # hours
        self.solar_constant = 1365.0  # W/m² (adjusted to max range)

        # Initialize scalers
        self.length_scale = self.earth_radius
        self.time_scale = self.day_period
        self.irradiance_scale = self.solar_constant

    def prepare_data(self,
                     lat,
                     lon,
                     time,
                     slope,
                     aspect,
                     atm,
                     cloud_cover=None,
                     wavelength=None):
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
            lat, lon, time_nd, slope, aspect, atm, cloud_cover, wavelength_nd
        ])
        return torch.FloatTensor(data)

    def normalize_data(self, data):
        """Normalize input data to [-1, 1] range"""
        # Define normalization ranges for each variable
        ranges = np.array([
            [-90, 90],  # latitude
            [-180, 180],  # longitude
            [0, 1],  # normalized time
            [0, 90],  # slope
            [0, 360],  # aspect
            [0, 1]  # atmospheric transmission
        ])

        # Normalize to [-1, 1]
        normalized = 2 * (data - ranges[:, 0]) / (ranges[:, 1] -
                                                  ranges[:, 0]) - 1
        return normalized

    def denormalize_predictions(self, predictions, scale_irradiance=True):
        """Convert predictions to physical units"""
        if scale_irradiance:
            # For irradiance predictions
            return torch.clamp(predictions * self.irradiance_scale, min=0)
        else:
            # For efficiency, apply hard clipping after sigmoid
            raw_efficiency = torch.sigmoid(predictions)
            return torch.clamp(0.15 + (0.10 * raw_efficiency), min=0.15, max=0.25)

    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data with enhanced edge cases and extreme conditions"""
        # Generate more diverse cases
        n_night = int(n_samples * 0.25)  # 25% nighttime cases
        n_transition = int(n_samples * 0.25)  # 25% sunrise/sunset
        n_extreme = int(n_samples * 0.1)  # 10% extreme conditions
        n_day = n_samples - n_night - n_transition - n_extreme

        # Nighttime data with more varied hours
        night_hours = np.concatenate([
            np.random.uniform(19, 24, n_night // 2),  # Late night
            np.random.uniform(0, 5, n_night // 2)     # Early morning
        ])

        # Enhanced transition data (sunrise/sunset)
        transition_hours = np.concatenate([
            np.random.uniform(5, 7, n_transition // 3),   # Early sunrise
            np.random.uniform(17, 19, n_transition // 3), # Sunset
            np.random.uniform(7, 17, n_transition // 3)   # Mixed conditions
        ])

        # Daytime data with peak hours
        day_hours = np.concatenate([
            np.random.uniform(10, 14, n_day // 2),    # Peak hours
            np.random.uniform(7, 17, n_day // 2)      # Regular day hours
        ])

        # Extreme condition hours (solar noon)
        extreme_hours = np.random.uniform(11, 13, n_extreme)

        # Combine all hours
        time = np.concatenate([night_hours, transition_hours, day_hours, extreme_hours])

        # Generate parameters with more extreme cases
        lat = np.concatenate([
            np.random.uniform(-90, -60, n_extreme),  # Polar regions
            np.random.uniform(60, 90, n_extreme),    # Polar regions
            np.random.uniform(-60, 60, n_samples - 2 * n_extreme)  # Regular latitudes
        ])

        lon = np.random.uniform(-180, 180, n_samples)
        
        # More varied slope angles
        slope = np.concatenate([
            np.random.uniform(0, 15, n_samples // 4),     # Low angles
            np.random.uniform(15, 45, n_samples // 2),    # Optimal range
            np.random.uniform(45, 90, n_samples // 4)     # High angles
        ])

        # Enhanced aspect variations
        aspect = np.concatenate([
            np.random.uniform(135, 225, n_samples // 2),  # Southern aspects
            np.random.uniform(0, 360, n_samples // 2)     # All directions
        ])

        # Atmospheric conditions with more extreme cases
        atm = np.concatenate([
            np.random.uniform(0.2, 0.5, n_samples // 4),  # Heavy atmosphere
            np.random.uniform(0.5, 0.8, n_samples // 2),  # Normal conditions
            np.random.uniform(0.8, 1.0, n_samples // 4)   # Clear sky
        ])

        # Enhanced cloud cover distribution
        cloud_cover = np.concatenate([
            np.zeros(n_samples // 4),                     # Clear sky
            np.random.uniform(0, 0.3, n_samples // 4),    # Light clouds
            np.random.uniform(0.3, 0.7, n_samples // 4),  # Moderate clouds
            np.random.uniform(0.7, 1.0, n_samples // 4)   # Heavy clouds
        ])

        # Wavelength variations
        wavelength = np.concatenate([
            np.random.uniform(0.3, 0.4, n_samples // 4),  # UV range
            np.random.uniform(0.4, 0.7, n_samples // 2),  # Visible light
            np.random.uniform(0.7, 1.0, n_samples // 4)   # Near IR
        ])

        return self.prepare_data(lat, lon, time, slope, aspect, atm,
                                 cloud_cover, wavelength)
