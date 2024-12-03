import numpy as np
import torch
import torch.nn as nn

class SolarPINN(nn.Module):
    def __init__(self, input_dim=11):  # Updated input dimension for new parameters
        super(SolarPINN, self).__init__()
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Physics coefficients
        self.solar_constant = 1367.0  # Solar constant (W/m²)
        self.ground_albedo = 0.2      # Ground albedo coefficient
        self.ref_temp = 25.0          # Reference temperature (°C)
        self.temp_coeff = -0.004      # Temperature coefficient (%/°C)
        
        # Spectral response parameters
        self.wavelength_peak = 550    # Peak response wavelength (nm)
        self.spectral_width = 100     # Response width (nm)

    def forward(self, x):
        raw_output = self.net(x)
        return torch.sigmoid(raw_output)

    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi * (284 + day_number) / 365))

    def hour_angle(self, time, lon):
        """Calculate hour angle (ω)"""
        hour = (time % 24).clamp(0, 24)
        return torch.deg2rad(15 * (hour - 12) + lon)

    def cos_incidence_angle(self, lat, lon, time, slope, aspect):
        """Calculate cosine of incidence angle (θ) and zenith angle"""
        lat_rad = torch.deg2rad(lat)
        slope_rad = torch.deg2rad(slope)
        aspect_rad = torch.deg2rad(aspect)

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

    def calculate_linke_turbidity(self, atm, cloud_cover):
        """Calculate Linke turbidity factor"""
        base_turbidity = 2.0 + 0.5 * atm
        cloud_effect = 0.5 + 2.0 * cloud_cover
        return base_turbidity * cloud_effect

    def calculate_air_mass(self, cos_zenith):
        """Calculate refined air mass using Kasten-Young formula"""
        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        return torch.where(
            cos_zenith > 0.001,
            1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364)),
            torch.tensor(float('inf'))
        )

    def hay_davies_diffuse(self, dni, cos_theta, cos_zenith, cloud_cover):
        """Calculate diffuse radiation using Hay-Davies model"""
        # Anisotropy index
        ai = torch.clamp(dni / self.solar_constant, min=0.0, max=1.0)
        
        # Circumsolar component
        circumsolar = ai * (cos_theta / cos_zenith)
        
        # Isotropic component
        isotropic = (1.0 - ai) * (1.0 + torch.cos(torch.deg2rad(slope))) / 2.0
        
        # Total diffuse
        diffuse_total = dni * (circumsolar + isotropic) * (1.0 - cloud_cover)
        return torch.clamp(diffuse_total, min=0.0)

    def calculate_cell_temperature(self, ambient_temp, irradiance):
        """Calculate cell temperature based on ambient temperature and irradiance"""
        noct = 45.0  # Nominal Operating Cell Temperature (°C)
        return ambient_temp + (noct - 20.0) * irradiance / 800.0

    def spectral_response(self, wavelength):
        """Calculate spectral response using Gaussian model"""
        return torch.exp(-((wavelength - self.wavelength_peak)**2) / 
                        (2 * self.spectral_width**2))

    def physics_loss(self, x, y_pred):
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength, ambient_temp, ground_type, humidity = (x[:, i] for i in range(11))

        cos_theta, cos_zenith = self.cos_incidence_angle(lat, lon, time, slope, aspect)
        
        # Atmospheric modeling
        linke_turbidity = self.calculate_linke_turbidity(atm, cloud_cover)
        air_mass = self.calculate_air_mass(cos_zenith)
        
        # Direct solar irradiance
        transmission = torch.exp(-linke_turbidity * air_mass / 20.0)
        dni = self.solar_constant * transmission * cos_zenith
        
        # Diffuse radiation
        diffuse = self.hay_davies_diffuse(dni, cos_theta, cos_zenith, cloud_cover)
        
        # Ground-reflected radiation
        ground_reflected = (dni + diffuse) * self.ground_albedo * (1.0 - torch.cos(torch.deg2rad(slope))) / 2.0
        
        # Total theoretical irradiance
        theoretical_irradiance = dni * cos_theta + diffuse + ground_reflected
        
        # Temperature effects
        cell_temp = self.calculate_cell_temperature(ambient_temp, theoretical_irradiance)
        temp_factor = 1.0 + self.temp_coeff * (cell_temp - self.ref_temp)
        
        # Spectral effects
        spectral_factor = self.spectral_response(wavelength)
        
        # Combined theoretical output
        theoretical_output = theoretical_irradiance * temp_factor * spectral_factor
        
        # Physics residuals
        nighttime_condition = (cos_zenith <= 0.001)
        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred) * 100.0,
            torch.zeros_like(y_pred)
        )
        
        physics_residual = torch.where(
            theoretical_output < 1.0,
            torch.abs(y_pred) * 50.0,
            (y_pred - theoretical_output/self.solar_constant)**2
        )
        
        # Efficiency constraints (15-25% range with temperature adjustment)
        efficiency_penalty = (
            torch.relu(0.15 * temp_factor - y_pred) + 
            torch.relu(y_pred - 0.25 * temp_factor)
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
        
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        total_loss = self.w_data * mse + self.w_physics * physics_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(), 0.0
