import numpy as np
import torch

class SolarPhysicsIdeal:
    def __init__(self):
        # Universal physical constants
        self.solar_constant = 1367.0  # Solar constant, S (W/m²)
        self.extinction_coefficient = 0.1  # Idealized extinction coefficient for clear sky

    def calculate_declination(self, day_of_year):
        """Calculate the solar declination angle (δ) for a given day of the year."""
        declination = 23.45 * torch.sin(2 * torch.pi * (day_of_year - 81) / 365)
        return torch.deg2rad(declination)

    def calculate_zenith_angle(self, latitude, declination, hour_angle):
        """Calculate the cosine of the solar zenith angle (cos θz)."""
        latitude_rad = torch.deg2rad(latitude)
        return (
            torch.sin(latitude_rad) * torch.sin(declination) +
            torch.cos(latitude_rad) * torch.cos(declination) * torch.cos(hour_angle)
        )

    def calculate_air_mass(self, zenith_angle):
        """Calculate air mass (M) using the Kasten and Young model."""
        zenith_angle_deg = torch.rad2deg(torch.acos(zenith_angle))
        air_mass = 1 / (torch.cos(torch.deg2rad(zenith_angle_deg)) + 
                    0.50572 * (96.07995 - zenith_angle_deg) ** -1.6364)
        return air_mass

    def calculate_atmospheric_transmission(self, air_mass):
        """Calculate the atmospheric transmission factor (T)."""
        return torch.exp(-self.extinction_coefficient * air_mass)

    def calculate_surface_orientation_factor(self, zenith_angle, slope, sun_azimuth, panel_azimuth):
        """Calculate the surface orientation factor (f_surf)."""
        slope_rad = torch.deg2rad(slope)
        sun_azimuth_rad = torch.deg2rad(sun_azimuth)
        panel_azimuth_rad = torch.deg2rad(panel_azimuth)

        return (
            torch.cos(slope_rad) * zenith_angle +
            torch.sin(slope_rad) * torch.sqrt(1 - zenith_angle ** 2) *
            torch.cos(sun_azimuth_rad - panel_azimuth_rad)
        )

    def calculate_hour_angle(self, time):
        """Calculate the hour angle (h) based on the time of day."""
        return torch.deg2rad(15 * (time - 12))

    def calculate_irradiance(self, latitude, time, slope=0, panel_azimuth=0):
        """Calculate the total solar irradiance at the surface under ideal conditions."""
        # Ensure inputs are tensors with proper dtype
        latitude = torch.as_tensor(latitude, dtype=torch.float32)
        time = torch.as_tensor(time, dtype=torch.float32)
        slope = torch.as_tensor(slope, dtype=torch.float32)
        panel_azimuth = torch.as_tensor(panel_azimuth, dtype=torch.float32)
        
        # Convert day of year from time
        day_of_year = torch.floor(time / 24 * 365)
        
        # Step 1: Calculate solar declination
        declination = self.calculate_declination(day_of_year)

        # Step 2: Calculate the hour angle
        hour_angle = self.calculate_hour_angle(time % 24)

        # Step 3: Calculate cosine of the solar zenith angle
        cos_zenith = self.calculate_zenith_angle(latitude, declination, hour_angle)
        cos_zenith = torch.clamp(cos_zenith, min=0.0)  # Ensure no negative values (nighttime)

        # Step 4: Calculate air mass
        air_mass = self.calculate_air_mass(cos_zenith)

        # Step 5: Calculate atmospheric transmission
        transmission = self.calculate_atmospheric_transmission(air_mass)

        # Step 6: Calculate surface orientation factor
        surface_orientation = self.calculate_surface_orientation_factor(
            cos_zenith, slope, hour_angle, panel_azimuth
        )

        # Step 7: Calculate irradiance
        irradiance = self.solar_constant * transmission * surface_orientation

        return torch.clamp(irradiance, min=0.0)  # Ensure non-negative irradiance

def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics between true and predicted values."""
    # Mean Absolute Error
    mae = torch.mean(torch.abs(y_true - y_pred))
    
    # Root Mean Square Error
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
    # R² Score
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item()
    }
