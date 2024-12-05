import torch
import torch.nn.functional as F

class EnhancedPINNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.physics_weight_scheduler = self.create_physics_weight_scheduler()

    def create_physics_weight_scheduler(self):
        """Create enhanced adaptive physics weight scheduler"""
        def scheduler(epoch):
            # Cosine annealing schedule for physics weight
            base_weight = 0.1
            max_weight = 2.0
            cycle_length = 10  # epochs per cycle
            
            # Calculate cycle progress
            cycle_progress = (epoch % cycle_length) / cycle_length
            
            # Cosine annealing formula
            weight = base_weight + (max_weight - base_weight) * \
                    (1 + np.cos(cycle_progress * np.pi)) / 2
                    
            # Add warm-up period
            if epoch < 5:
                weight *= epoch / 5
                
            return weight
        return scheduler

    def train_step(self, x_data, y_data, epoch):
        self.optimizer.zero_grad()

        # Forward pass with gradient clipping
        y_pred = self.model(x_data)

        # Multi-scale loss calculation
        data_loss = 0.5 * F.mse_loss(y_pred, y_data) + \
                   0.3 * F.l1_loss(y_pred, y_data) + \
                   0.2 * self.huber_loss(y_pred, y_data)
                   
        # Enhanced physics loss with multiple components
        physics_loss = self.model.physics_loss(x_data, y_pred)
        conservation_loss = self.model.energy_conservation(x_data, y_pred)
        spectral_loss = self.model.spectral_constraints(x_data, y_pred)

        # Adaptive weighting
        physics_weight = self.physics_weight_scheduler(epoch)
        total_loss = data_loss + \
                    physics_weight * (0.5 * physics_loss + 
                                    0.3 * conservation_loss +
                                    0.2 * spectral_loss)

        # Gradient clipping and backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'conservation_loss': conservation_loss.item(),
            'spectral_loss': spectral_loss.item()
        }
        
    def huber_loss(self, pred, target, delta=0.1):
        """Huber loss for robust training"""
        return F.smooth_l1_loss(pred, target, beta=delta)
