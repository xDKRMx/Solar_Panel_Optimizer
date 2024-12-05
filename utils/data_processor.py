import numpy as np
import torch


class DataProcessor:

    def __init__(self):
        # Physical constants for non-dimensionalization
        self.earth_radius = 6371.0  # km
        self.day_period = 24.0  # hours
        self.solar_constant = 1367.0  # W/m²
        self.extinction_coefficient = 0.1  # Atmospheric extinction coefficient

        # Initialize scalers
        self.length_scale = self.earth_radius
        self.time_scale = self.day_period
        self.irradiance_scale = self.solar_constant

        # Terrain parameters
        self.max_slope = 45.0  # Maximum terrain slope in degrees
        self.terrain_roughness = 0.3  # Terrain roughness factor

    def generate_training_data(self, n_points=1000):
        """Generate physics-based training data for PINN"""
        return self.generate_physics_based_data(n_points)

    def generate_physics_based_data(self, n_points):
        """Generate data points based on physical constraints"""
        # Structured grid for spatial coordinates
        lat_space = np.linspace(-90, 90, int(np.sqrt(n_points)))
        lon_space = np.linspace(-180, 180, int(np.sqrt(n_points)))
        lat_grid, lon_grid = np.meshgrid(lat_space, lon_space)

        # Time sampling based on solar physics
        time_points = self.generate_solar_time_points(n_points)

        # Calculate physically meaningful parameters
        slope = self.calculate_terrain_slope(lat_grid, lon_grid)
        aspect = self.calculate_terrain_aspect(lat_grid, lon_grid)
        atm_transmission = self.calculate_atmospheric_transmission(
            lat_grid, time_points
        )

        return self.prepare_physics_data(
            lat_grid.flatten(),
            lon_grid.flatten(),
            time_points,
            slope.flatten(),
            aspect.flatten(),
            atm_transmission.flatten()
        )

    def calculate_atmospheric_transmission(self, lat, time):
        """Physical atmospheric transmission model"""
        # Air mass calculation
        air_mass = 1 / (np.cos(np.deg2rad(lat)) + 0.50572 * 
                    (96.07995 - lat)**(-1.6364))

        # Beer-Lambert law
        return np.exp(-self.extinction_coefficient * air_mass)

    def calculate_terrain_slope(self, lat_grid, lon_grid):
        """Calculate terrain slope based on location"""
        # Simplified terrain model using latitude dependency
        base_slope = self.max_slope * np.abs(lat_grid) / 90.0
        
        # Add terrain roughness variation
        random_variation = np.random.normal(0, self.terrain_roughness, lat_grid.shape)
        slope = base_slope + random_variation
        
        return np.clip(slope, 0, self.max_slope)

    def calculate_terrain_aspect(self, lat_grid, lon_grid):
        """Calculate terrain aspect (orientation) based on location"""
        # Calculate aspect based on latitude and longitude gradients
        dy = np.gradient(lat_grid, axis=0)
        dx = np.gradient(lon_grid, axis=1)
        
        # Convert to compass bearing (0-360 degrees)
        aspect = np.rad2deg(np.arctan2(dy, dx)) % 360
        return aspect

    def generate_solar_time_points(self, n_points):
        """Generate time points optimized for solar position sampling"""
        # Generate more samples around sunrise/sunset
        morning = np.linspace(6, 8, n_points // 4)  # Dawn
        day = np.linspace(8, 16, n_points // 2)     # Day
        evening = np.linspace(16, 18, n_points // 4) # Dusk
        
        return np.concatenate([morning, day, evening])

    def prepare_physics_data(self, lat, lon, time, slope, aspect, atm):
        """Prepare physics-based data for PINN model"""
        # Set default values for cloud cover and wavelength
        cloud_cover = np.zeros_like(lat)
        wavelength = np.full_like(lat, 0.5)  # Default wavelength in μm

        # Prepare data using the existing method
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
        normalized = 2 * (data - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1
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
            
    def prepare_spectral_data(self, x):
        """Add spectral components to the input data"""
        wavelength = x.split(1, dim=1)[-1]
        # Physical constants
        h = 6.626e-34  # Planck constant
        c = 2.998e8    # Speed of light
        k = 1.381e-23  # Boltzmann constant
        T = 5778       # Solar surface temperature
        
        spectral_irradiance = (2*h*c*c)/(wavelength**5) * \
            1/(torch.exp(h*c/(wavelength*k*T)) - 1)
        return spectral_irradiance