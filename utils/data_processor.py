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

    def generate_training_data(self, n_points=1000, n_time_points=24, n_angles=30, n_atm_points=10):
        """Generate structured grid training data for PINN"""
        # Create structured grids for spatial coordinates
        lat_space = np.linspace(-90, 90, int(np.sqrt(n_points)))
        lon_space = np.linspace(-180, 180, int(np.sqrt(n_points)))
        
        # Create structured grids for time and angles
        time_space = np.linspace(0, 24, n_time_points)  # Full day coverage
        slope_space = np.linspace(0, 90, n_angles)  # Panel tilt angles
        aspect_space = np.linspace(0, 360, n_angles)  # Panel orientation angles
        
        # Create structured grids for atmospheric parameters
        atm_space = np.linspace(0.5, 1.0, n_atm_points)  # Atmospheric transmission
        cloud_space = np.linspace(0, 1.0, n_atm_points)  # Cloud cover
        wavelength_space = np.linspace(0.3, 1.0, n_atm_points)  # Solar spectrum
        
        # Create meshgrid combinations
        lat_grid, lon_grid, time_grid, slope_grid, aspect_grid, atm_grid, cloud_grid, wavelength_grid = np.meshgrid(
            lat_space, lon_space, time_space, slope_space, aspect_space,
            atm_space, cloud_space, wavelength_space,
            indexing='ij'
        )
        
        # Flatten all grids to 1D arrays
        lat = lat_grid.flatten()
        lon = lon_grid.flatten()
        time = time_grid.flatten()
        slope = slope_grid.flatten()
        aspect = aspect_grid.flatten()
        atm = atm_grid.flatten()
        cloud_cover = cloud_grid.flatten()
        wavelength = wavelength_grid.flatten()

        # Return prepared data with structured grid samples
        return self.prepare_data(
            lat=lat,
            lon=lon,
            time=time,
            slope=slope,
            aspect=aspect,
            atm=atm,
            cloud_cover=cloud_cover,
            wavelength=wavelength
        )
