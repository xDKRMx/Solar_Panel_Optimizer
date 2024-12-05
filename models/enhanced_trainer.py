import torch
import torch.nn.functional as F

class EnhancedPINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.physics_weight_scheduler = self.create_physics_weight_scheduler()

    def create_physics_weight_scheduler(self):
        """Create adaptive physics weight scheduler"""
        def scheduler(epoch):
            base_weight = 0.1
            max_weight = 1.0
            return min(base_weight + 0.1 * epoch, max_weight)
        return scheduler

    def train_step(self, x_data, y_data, epoch):
        self.optimizer.zero_grad()

        # Forward pass
        y_pred = self.model(x_data)

        # Calculate losses
        data_loss = F.mse_loss(y_pred, y_data)
        physics_loss = self.model.physics_loss(x_data, y_pred)

        # Adaptive physics weight
        physics_weight = self.physics_weight_scheduler(epoch)
        total_loss = data_loss + physics_weight * physics_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item()
        }
