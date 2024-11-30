import torch
import torch.nn as nn

class SolarPINN(nn.Module):
    def __init__(self, input_dim=6):
        super(SolarPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def physics_loss(self, x, y_pred):
        # Extract parameters from input
        lat, lon, time, slope, aspect, atm = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        
        # Compute gradients for physics residual
        y_grad = torch.autograd.grad(
            y_pred.sum(), x, 
            create_graph=True, retain_graph=True
        )[0]
        
        # Physics residual based on advection-diffusion equation
        residual = (y_grad[:, 2] + # Time derivative
                   y_grad[:, 0] + y_grad[:, 1] + # Spatial derivatives
                   0.1 * y_pred) # Source term
        
        return torch.mean(residual ** 2)

class PINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, x_data, y_data):
        self.optimizer.zero_grad()
        
        # Forward pass
        y_pred = self.model(x_data)
        
        # Compute losses
        mse = self.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)
        
        # Total loss
        total_loss = mse + 0.1 * physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), mse.item(), physics_loss.item()
