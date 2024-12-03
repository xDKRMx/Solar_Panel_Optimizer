import numpy as np
import torch
import torch.nn as nn

class SolarPINN(nn.Module):
    # Characteristic values for non-dimensionalization
    TIME_CHAR = 24.0  # hours
    IRRADIANCE_CHAR = 1367.0  # W/m²
    TEMP_CHAR = 298.15  # K
    ANGLE_CHAR = 360.0  # degrees
    WAVELENGTH_CHAR = 750.0  # nm
    AIR_MASS_CHAR = 1.5  # Standard air mass reference (AM1.5)
    BETA_CHAR = 0.5  # Maximum expected turbidity
    CLOUD_CHAR = 1.0  # Maximum cloud cover
    DISTANCE_CHAR = 1.0  # meters
    PRESSURE_CHAR = 101325.0  # Pascal (1 atm)
    POWER_CHAR = 1000.0  # Watts
    VELOCITY_CHAR = 10.0  # m/s (typical max wind speed)
    HUMIDITY_CHAR = 100.0  # percent
    
    def __init__(self, input_dim=12):  # Updated input dimension for new parameters
        super(SolarPINN, self).__init__()
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights using Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Essential physics coefficients
        self.solar_constant = 1367.0  # Solar constant (W/m²)
        self.beta = 0.125      # Atmospheric turbidity coefficient
        self.ref_temp = 25.0   # Reference temperature (°C)
        self.ref_pressure = 101325.0  # Reference pressure (Pa)
    
    def normalize_inputs(self, x):
        """Normalize input features to [0,1] range"""
        # Unpack input features
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength, \
        panel_width, panel_height, wind_speed, humidity = (x[:, i] for i in range(12))
        
        # Normalize each feature
        norm_lat = (lat + 90) / 180  # [-90, 90] -> [0, 1]
        norm_lon = (lon + 180) / 360  # [-180, 180] -> [0, 1]
        norm_time = time / self.TIME_CHAR  # [0, 24] -> [0, 1]
        norm_slope = slope / 90  # [0, 90] -> [0, 1]
        norm_aspect = (aspect + 180) / self.ANGLE_CHAR  # [-180, 180] -> [0, 1]
        norm_atm = atm / self.BETA_CHAR  # Normalize atmospheric turbidity
        norm_cloud = cloud_cover / self.CLOUD_CHAR  # Normalize cloud cover
        norm_wavelength = wavelength / self.WAVELENGTH_CHAR  # [0, 750] -> [0, 1]
        norm_width = panel_width / self.DISTANCE_CHAR  # Normalize panel width
        norm_height = panel_height / self.DISTANCE_CHAR  # Normalize panel height
        norm_wind = wind_speed / self.VELOCITY_CHAR  # Normalize wind speed
        norm_humidity = humidity / self.HUMIDITY_CHAR  # Normalize humidity
        
        return torch.stack([
            norm_lat, norm_lon, norm_time, norm_slope, norm_aspect,
            norm_atm, norm_cloud, norm_wavelength, norm_width, norm_height,
            norm_wind, norm_humidity
        ], dim=1)

    def denormalize_outputs(self, y, panel_width, panel_height):
        """Convert normalized predictions to physical power output"""
        # Calculate panel area
        panel_area = (panel_width * panel_height) / (self.DISTANCE_CHAR ** 2)
        
        # Convert normalized irradiance to power output
        irradiance = y * self.IRRADIANCE_CHAR
        power_output = irradiance * panel_area  # W/m² * m² = W
        
        return power_output / self.POWER_CHAR  # Normalize to characteristic power

    def forward(self, x):
        # Normalize inputs
        x_norm = self.normalize_inputs(x)
        
        # Pass through network
        raw_output = self.net(x_norm)
        
        # Normalize to [0,1] range using sigmoid
        y_norm = torch.sigmoid(raw_output)
        
        # Get panel dimensions for power calculation
        panel_width = x[:, 8]
        panel_height = x[:, 9]
        
        # Denormalize to physical units with power calculation
        return self.denormalize_outputs(y_norm, panel_width, panel_height)

    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        time_phys = time * self.TIME_CHAR
        day_number = (time_phys / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        time_phys = time * self.TIME_CHAR
        hour = (time_phys % 24).clamp(0, 24)
        return torch.deg2rad(15 * (hour - 12) + lon * self.ANGLE_CHAR - 180)

    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ) and zenith angle"""
        lat_phys = (lat * 180) - 90
        lon_phys = (lon * 360) - 180
        slope_phys = slope * 90
        aspect_phys = (aspect * self.ANGLE_CHAR) - 180
        
        lat_rad = torch.deg2rad(lat_phys)
        slope_rad = torch.deg2rad(slope_phys)
        aspect_rad = torch.deg2rad(aspect_phys)

        declination = self.solar_declination(time)
        hour_angle = self.hour_angle(time, lon)

        cos_zenith = (torch.sin(lat_rad) * torch.sin(declination) +
                     torch.cos(lat_rad) * torch.cos(declination) *
                     torch.cos(hour_angle))

        cos_theta = (
            torch.sin(lat_rad) * torch.sin(declination) * torch.cos(slope_rad)
            - torch.cos(lat_rad) * torch.sin(declination) *
            torch.sin(slope_rad) * torch.cos(aspect_rad) +
            torch.cos(lat_rad) * torch.cos(declination) *
            torch.cos(hour_angle) * torch.cos(slope_rad) +
            torch.sin(lat_rad) * torch.cos(declination) * torch.cos(hour_angle)
            * torch.sin(slope_rad) * torch.cos(aspect_rad) +
            torch.cos(declination) * torch.sin(hour_angle) *
            torch.sin(slope_rad) * torch.sin(aspect_rad))

        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)

    def physics_loss(self, x, y_pred):
        # Normalize input features
        x_norm = self.normalize_inputs(x)
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength, \
        panel_width, panel_height, wind_speed, humidity = (x_norm[:, i] for i in range(12))
        
        # Get angles using normalized inputs
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)

        # Calculate normalized air mass with pressure correction
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        pressure_ratio = self.ref_pressure / self.PRESSURE_CHAR
        air_mass = pressure_ratio * (1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364)))
        normalized_air_mass = air_mass / self.AIR_MASS_CHAR

        # Simple day/night validation
        nighttime_condition = (cos_zenith <= 0.001)
        y_pred_norm = y_pred / self.POWER_CHAR  # Normalize predictions
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred_norm) * 100.0,
            torch.zeros_like(y_pred_norm)
        )

        # Normalize atmospheric parameters
        normalized_beta = self.beta / self.BETA_CHAR
        normalized_cloud = cloud_cover / self.CLOUD_CHAR
        normalized_humidity = humidity  # Already normalized
        
        # Calculate transmission with normalized values and humidity effect
        humidity_factor = 1.0 - 0.1 * normalized_humidity  # Reduce transmission with humidity
        transmission = torch.exp(-normalized_beta * normalized_air_mass) * humidity_factor
        
        # Calculate theoretical power output
        theoretical_irradiance = (self.solar_constant / self.IRRADIANCE_CHAR) * transmission * cos_theta
        theoretical_irradiance = theoretical_irradiance * (1 - 0.75 * normalized_cloud ** 3.4)
        
        # Calculate panel area and normalize
        panel_area = (panel_width * panel_height) / (self.DISTANCE_CHAR ** 2)
        theoretical_power = theoretical_irradiance * panel_area / self.POWER_CHAR
        
        # Physics residual based on normalized power values
        physics_residual = torch.where(
            theoretical_power < 0.001,
            torch.abs(y_pred_norm) * 50.0,
            (y_pred_norm - theoretical_power)**2
        )

        # Temperature and wind speed effects
        normalized_temp = (20.0 - self.ref_temp) / self.TEMP_CHAR
        wind_cooling = 0.1 * wind_speed  # Wind speed effect on temperature
        temp_effect = 1.0 - 0.005 * (normalized_temp * self.TEMP_CHAR - wind_cooling)
        
        # Efficiency constraints (15-25% range)
        efficiency_norm = y_pred_norm / theoretical_power.clamp(min=0.001)
        efficiency_penalty = (
            torch.relu(0.15 - efficiency_norm) + 
            torch.relu(efficiency_norm - 0.25)
        ) * 100.0

        total_residual = (
            5.0 * physics_residual +
            10.0 * nighttime_penalty +
            2.0 * efficiency_penalty
        )

        return torch.mean(total_residual)

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999)
        )
        self.mse_loss = nn.MSELoss()
        
        # Loss weights
        self.w_data = 0.3
        self.w_physics = 0.7

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        y_pred = self.model(x_data)
        
        # Get panel dimensions for power normalization
        panel_width = x_data[:, 8]
        panel_height = x_data[:, 9]
        
        # Normalize targets and predictions for loss calculation
        y_data_norm = self.model.denormalize_outputs(y_data / self.model.IRRADIANCE_CHAR, panel_width, panel_height)
        y_pred_norm = y_pred  # Already normalized to power units
        
        mse = self.mse_loss(y_pred_norm, y_data_norm)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        total_loss = self.w_data * mse + self.w_physics * physics_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(), 0.0
