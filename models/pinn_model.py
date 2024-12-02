import numpy as np
import torch
import torch.nn as nn

class ConfigurableBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.linear(x))))

class SolarPINN(nn.Module):
    def __init__(self, input_dim=8, complexity_factor=1.0):
        super(SolarPINN, self).__init__()
        # Scale layer sizes based on input complexity
        base_width = int(96 * complexity_factor)
        
        # Define network layers with skip connections
        self.input_block = ConfigurableBlock(input_dim, base_width)
        self.hidden1 = ConfigurableBlock(base_width, base_width * 2)
        self.hidden2 = ConfigurableBlock(base_width * 2, base_width * 2)
        self.hidden3 = ConfigurableBlock(base_width * 2, base_width)
        self.output = nn.Linear(base_width + input_dim, 1)  # Skip connection from input
        
        # Initialize weights
        self._init_weights()
        
        # Physics coefficients (now as trainable parameters)
        self.register_parameter('solar_constant', nn.Parameter(torch.tensor(1367.0)))
        self.register_parameter('ref_wavelength', nn.Parameter(torch.tensor(0.55)))
        self.register_parameter('beta', nn.Parameter(torch.tensor(0.125)))
        self.register_parameter('alpha', nn.Parameter(torch.tensor(1.25)))
        self.register_parameter('cloud_alpha', nn.Parameter(torch.tensor(0.82)))
        self.register_parameter('ref_temp', nn.Parameter(torch.tensor(25.0)))
        self.register_parameter('temp_coeff', nn.Parameter(torch.tensor(0.0045)))
        
        # Parameter constraints
        self.param_ranges = {
            'solar_constant': (1360.0, 1380.0),
            'ref_wavelength': (0.5, 0.6),
            'beta': (0.1, 0.15),
            'alpha': (1.0, 1.5),
            'cloud_alpha': (0.7, 0.9),
            'ref_temp': (20.0, 30.0),
            'temp_coeff': (0.003, 0.006)
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _constrain_parameters(self):
        for name, param in self.named_parameters():
            if name in self.param_ranges:
                min_val, max_val = self.param_ranges[name]
                with torch.no_grad():
                    param.clamp_(min_val, max_val)

    def forward(self, x):
        # Input normalization
        x_normalized = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
        
        # Forward pass with skip connections
        h1 = self.input_block(x_normalized)
        h2 = self.hidden1(h1)
        h3 = self.hidden2(h2)
        h4 = self.hidden3(h3)
        
        # Concatenate input with final hidden layer for skip connection
        combined = torch.cat([h4, x_normalized], dim=1)
        raw_output = self.output(combined)
        
        # Apply sigmoid for output scaling
        output = torch.sigmoid(raw_output)
        
        # Constrain physics parameters
        self._constrain_parameters()
        
        return output

    def solar_declination(self, time):
        """Calculate solar declination angle (δ)"""
        day_number = (time / 24.0 * 365).clamp(0, 365)
        return torch.deg2rad(23.45 * torch.sin(2 * np.pi *
                                               (284 + day_number) / 365))

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

        return torch.clamp(cos_theta, min=0.0), torch.clamp(
            cos_zenith, min=0.0001)  

    def calculate_optical_depth(self,
                                wavelength,
                                air_mass,
                                altitude=0,
                                cloud_cover=0):
        """Calculate advanced multi-wavelength optical depth with cloud and altitude effects"""

        base_depth = self.beta * (wavelength /
                                  self.ref_wavelength)**(-self.alpha)

        altitude_factor = torch.exp(-altitude / 7.4)  

   
        cloud_factor = 1.0 + (cloud_cover**2) * (1.0 - 0.5 * altitude_factor)

   
        air_mass_factor = torch.exp(-base_depth * air_mass * altitude_factor *
                                    cloud_factor)

        cloud_scatter = 0.2 * cloud_cover * (wavelength /
                                             self.ref_wavelength)**(-0.75)

        return base_depth * air_mass_factor + cloud_scatter

    def physics_loss(self, x, y_pred):
        lat, lon, time, slope, aspect, atm, cloud_cover, wavelength = (
            x[:, i] for i in range(8))

        cos_theta, cos_zenith = self.cos_incidence_angle(
            lat, lon, time, slope, aspect)

        nighttime_threshold = 0.001
        twilight_threshold = 0.1
        nighttime_condition = (cos_zenith <= nighttime_threshold)
        twilight_condition = (cos_zenith > nighttime_threshold) & (cos_zenith <= twilight_threshold)
        

        nighttime_penalty = torch.where(
            nighttime_condition,
            torch.abs(y_pred) * 1000.0, 
            torch.where(
                twilight_condition,
                torch.abs(y_pred - self.solar_constant * cos_zenith) * 100.0,  
                torch.zeros_like(y_pred)
            ))

        y_grad = torch.autograd.grad(y_pred.sum(),
                                   x,
                                   create_graph=True,
                                   retain_graph=True)[0]

        zenith_angle = torch.acos(cos_zenith)
        zenith_deg = torch.rad2deg(zenith_angle)
        air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - zenith_deg).pow(-1.6364))

        optical_depth = self.calculate_optical_depth(wavelength,
                                                   air_mass,
                                                   altitude=0,
                                                   cloud_cover=cloud_cover)

        base_transmission = 1.0 - self.cloud_alpha * (cloud_cover**2)
        diffuse_factor = 0.3 * cloud_cover * (1.0 - cos_zenith)
        cloud_transmission = base_transmission + diffuse_factor

        y_grad2 = torch.autograd.grad(y_grad.sum(),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)[0]

        flux_divergence = y_grad2[:, 0] + y_grad2[:, 1]
        source_term = y_grad[:, 2]
        conservation_residual = flux_divergence + source_term

        direct_irradiance = (self.solar_constant * 
                           torch.exp(-optical_depth * air_mass) * 
                           cos_theta * cloud_transmission)
        
        diffuse_factor = 0.3 + 0.7 * cloud_cover  
        diffuse_irradiance = (self.solar_constant * diffuse_factor * 
                            (1.0 - cloud_transmission) * cos_zenith *
                            torch.exp(-optical_depth * air_mass * 0.5))  
        

        ground_albedo = 0.2 + 0.1 * cloud_cover  
        reflected_irradiance = (ground_albedo * (direct_irradiance + diffuse_irradiance) * 
                              (1.0 - cos_theta) * 0.5)

        
        cos_zenith = torch.clamp(cos_zenith, min=0.001, max=1.0)
        theoretical_irradiance = torch.where(
            cos_zenith > 0,
            direct_irradiance + diffuse_irradiance + reflected_irradiance,
            torch.zeros_like(direct_irradiance))

        
        nighttime_penalty = torch.where(
            cos_zenith <= 0.001,
            torch.abs(y_pred) * 1000.0, 
            torch.zeros_like(y_pred)
        )

        
        spatial_residual = y_grad[:, 0]**2 + y_grad[:, 1]**2
        temporal_residual = y_grad[:, 2]

       
        physics_residual = torch.where(
            theoretical_irradiance < 1.0,  
            torch.abs(y_pred) * 500.0,    
            (y_pred - theoretical_irradiance)**2
        )

       
        spatial_weight = 0.2 * torch.exp(-conservation_residual.abs().mean())
        temporal_weight = 0.2 * torch.exp(-temporal_residual.abs().mean())
        boundary_weight = 0.15
        conservation_weight = 0.15

        
        efficiency_min = 0.15  
        efficiency_max = 0.25  
        
        
        efficiency_lower = torch.exp(-100 * (y_pred - efficiency_min))
        efficiency_upper = torch.exp(100 * (y_pred - efficiency_max))
        
       
        efficiency_penalty = (efficiency_lower + efficiency_upper) * 2000.0
        
        
        additional_lower_barrier = torch.exp(-200 * (y_pred - efficiency_min))
        additional_upper_barrier = torch.exp(200 * (y_pred - efficiency_max))
        efficiency_penalty += (additional_lower_barrier + additional_upper_barrier) * 5000.0

        
        clipping_penalty = torch.mean(torch.abs(
            y_pred - torch.clamp(y_pred, min=0.15, max=0.25)
        )) * 100.0

        
        # Enhanced residual calculation with adaptive weighting
        physics_scale = torch.exp(-physics_residual.mean()).detach()
        spatial_scale = torch.exp(-spatial_residual.mean()).detach()
        temporal_scale = torch.exp(-temporal_residual.mean()).detach()
        
        total_residual = (
            spatial_weight * spatial_residual * spatial_scale +
            temporal_weight * temporal_residual * temporal_scale +
            6.0 * physics_residual * physics_scale +
            boundary_weight * (torch.relu(-y_pred) + torch.relu(y_pred - self.solar_constant)) +
            conservation_weight * conservation_residual**2 +
            12.0 * nighttime_penalty +
            1.5 * efficiency_penalty +
            2.0 * clipping_penalty
        )

        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return torch.mean(total_residual)


class PINNTrainer:
    def __init__(self, model, learning_rate=0.001, patience=10):
        self.model = model
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Setup optimizer with cosine annealing scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # Initialize adaptive loss weights as parameters
        self.w_data = nn.Parameter(torch.tensor(0.35))
        self.w_physics = nn.Parameter(torch.tensor(0.45))
        self.w_boundary = nn.Parameter(torch.tensor(0.20))
        
        # Gradient clipping parameters
        self.grad_clip_threshold = 1.0
        self.adaptive_clip_factor = 1.0  

    def boundary_loss(self, x_data, y_pred):
        """Compute boundary condition losses"""
        
        night_loss = torch.mean(torch.relu(-y_pred))

       
        max_loss = torch.mean(torch.relu(y_pred - self.model.solar_constant))

        
        time = x_data[:, 2]
        time_start = self.model(x_data.clone().detach().requires_grad_(True))
        time_end = self.model(
            torch.cat([x_data[:, :2], (time + 24).unsqueeze(1), x_data[:, 3:]],
                      dim=1))
        periodicity_loss = torch.mean((time_start - time_end)**2)

        return night_loss + max_loss + periodicity_loss

    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()

        
        y_pred = self.model(x_data)

        
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        boundary_loss = self.boundary_loss(x_data, y_pred)

       
        total_loss = (self.w_data * mse + self.w_physics * physics_loss +
                      self.w_boundary * boundary_loss)

        
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), mse.item(), physics_loss.item(
        ), boundary_loss.item()