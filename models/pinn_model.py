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
    DAY_CHAR = 365.0  # days in year
    HOUR_ANGLE_CHAR = 15.0  # degrees per hour
    EFFICIENCY_CHAR = 1.0  # maximum efficiency
    DECLINATION_CHAR = 23.45  # maximum declination angle
    ZENITH_PENALTY_CHAR = 100.0  # maximum penalty value
    TEMP_COEF_CHAR = 0.005  # temperature coefficient
    
    def __init__(self, input_dim=8): 
        super(SolarPINN, self).__init__()
        # Simplified network architecture with 3 layers
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
    
    def normalize_inputs(self, x):
        """Normalize input features to [0,1] range"""
        # Unpack input features
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (x[:, i] for i in range(8))
        
        # Normalize each feature
        norm_lat = (lat + 90) / 180  # [-90, 90] -> [0, 1]
        norm_lon = (lon + 180) / 360  # [-180, 180] -> [0, 1]
        norm_time = time / self.TIME_CHAR  # [0, 24] -> [0, 1]
        norm_slope = slope / 90  # [0, 90] -> [0, 1]
        norm_aspect = (aspect + 180) / self.ANGLE_CHAR  # [-180, 180] -> [0, 1]
        norm_atm = atm / self.BETA_CHAR  # Normalize atmospheric turbidity
        norm_cloud = cloud_cover / self.CLOUD_CHAR  # Normalize cloud cover
        norm_wavelength = wavelength / self.WAVELENGTH_CHAR  # [0, 750] -> [0, 1]
        
        return torch.stack([
            norm_lat, norm_lon, norm_time, norm_slope, norm_aspect,
            norm_atm, norm_cloud, norm_wavelength
        ], dim=1)

    def denormalize_outputs(self, y):
        """Convert normalized predictions back to physical units"""
        return y * self.IRRADIANCE_CHAR

    def forward(self, x):
        # Normalize inputs
        x_norm = self.normalize_inputs(x)
        
        # Pass through network
        raw_output = self.net(x_norm)
        
        # Normalize to [0,1] range using sigmoid
        y_norm = torch.sigmoid(raw_output)
        
        # Denormalize to physical units
        return self.denormalize_outputs(y_norm)

    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        # Convert normalized time to normalized day number
        time_phys = time * self.TIME_CHAR
        day_number = (time_phys / 24.0 * self.DAY_CHAR).clamp(0, self.DAY_CHAR)
        
        # Normalize constants in calculation
        norm_declination = self.DECLINATION_CHAR / self.ANGLE_CHAR
        norm_day_offset = 284.0 / self.DAY_CHAR
        
        return torch.deg2rad(
            self.DECLINATION_CHAR * torch.sin(
                2 * np.pi * (norm_day_offset + day_number / self.DAY_CHAR)
            )
        )

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        # Convert normalized time to normalized hours
        time_phys = time * self.TIME_CHAR
        hour = (time_phys % self.TIME_CHAR).clamp(0, self.TIME_CHAR)
        
        # Normalize hour angle calculation
        norm_hour_rate = self.HOUR_ANGLE_CHAR / self.ANGLE_CHAR
        norm_time_offset = 12.0 / self.TIME_CHAR
        
        return torch.deg2rad(
            self.HOUR_ANGLE_CHAR * (hour / self.TIME_CHAR - norm_time_offset) +
            lon * self.ANGLE_CHAR - 180
        )

    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ) and zenith angle"""
        # Convert normalized values to physical units for angle calculations
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
            torch.sin(slope_rad) * torch.sin(aspect_rad)
        )

        return torch.clamp(cos_theta, min=0.0), torch.clamp(cos_zenith, min=0.0001)

    def physics_loss(self, x, y_pred):
        # Normalize input features
        x_norm = self.normalize_inputs(x)
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (x_norm[:, i] for i in range(8))
        
        # Get angles using normalized inputs
        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)

        # Calculate normalized air mass with normalized constants
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        
        # Normalize air mass formula constants
        AM_CONST1 = 0.50572 / self.AIR_MASS_CHAR
        AM_CONST2 = 96.07995 / 90.0  # Normalize to angle range
        AM_CONST3 = -1.6364  # Dimensionless exponent
        
        air_mass = 1.0 / (cos_zenith + AM_CONST1 * 
                         (AM_CONST2 * (90.0 - zenith_deg)).pow(AM_CONST3))
        normalized_air_mass = air_mass / self.AIR_MASS_CHAR

        # Simple day/night validation with normalized penalty
        nighttime_condition = (cos_zenith <= 0.001)
        y_pred_norm = y_pred / self.IRRADIANCE_CHAR  # Normalize predictions
        night_penalty_factor = self.ZENITH_PENALTY_CHAR / self.EFFICIENCY_CHAR
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred_norm) * night_penalty_factor,
            torch.zeros_like(y_pred_norm)
        )

        # Normalize atmospheric parameters
        normalized_beta = self.beta / self.BETA_CHAR
        normalized_cloud = cloud_cover / self.CLOUD_CHAR
        
        # Calculate transmission with normalized values
        transmission = torch.exp(-normalized_beta * normalized_air_mass)
        theoretical_irradiance = (self.solar_constant / self.IRRADIANCE_CHAR) * transmission * cos_theta
        
        # Normalize cloud effect constants
        CLOUD_COEF = 0.75 / self.CLOUD_CHAR
        CLOUD_EXP = 3.4  # Dimensionless exponent
        theoretical_irradiance = theoretical_irradiance * (1 - CLOUD_COEF * normalized_cloud ** CLOUD_EXP)
        
        # Physics residual based on normalized values
        physics_penalty_factor = 50.0 / self.ZENITH_PENALTY_CHAR
        physics_residual = torch.where(
            theoretical_irradiance < 0.001,
            torch.abs(y_pred_norm) * physics_penalty_factor,
            (y_pred_norm - theoretical_irradiance)**2
        )

        # Temperature effects with normalized coefficients
        normalized_temp = (20.0 - self.ref_temp) / self.TEMP_CHAR
        temp_coef_norm = self.TEMP_COEF_CHAR / self.EFFICIENCY_CHAR
        temp_effect = 1.0 - temp_coef_norm * (normalized_temp * self.TEMP_CHAR)
        
        # Efficiency constraints with normalized bounds
        efficiency_norm = y_pred_norm / theoretical_irradiance.clamp(min=0.001)
        EFF_MIN = 0.15 / self.EFFICIENCY_CHAR
        EFF_MAX = 0.25 / self.EFFICIENCY_CHAR
        efficiency_penalty = (
            torch.relu(EFF_MIN - efficiency_norm) + 
            torch.relu(efficiency_norm - EFF_MAX)
        ) * (self.ZENITH_PENALTY_CHAR / self.EFFICIENCY_CHAR)

        # Combine losses with normalized weights
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
        
        # Normalize targets for loss calculation
        y_data_norm = y_data / self.model.IRRADIANCE_CHAR
        y_pred_norm = y_pred / self.model.IRRADIANCE_CHAR
        
        mse = self.mse_loss(y_pred_norm, y_data_norm)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        total_loss = self.w_data * mse + self.w_physics * physics_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(), 0.0
