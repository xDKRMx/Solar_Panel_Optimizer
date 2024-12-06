import torch
import numpy as np
import torch.nn.functional as F
from solar_pinn_ideal import SolarPINN, PINNTrainer
from physics_validator import SolarPhysicsIdeal

def generate_training_data(n_samples=2000, validation_split=0.2):
    """Generate synthetic training data with enhanced equatorial sampling."""
    # Calculate sample distribution
    equatorial_samples = int(n_samples * 0.5)  # 50% samples for -45° to +45°
    polar_samples = n_samples - equatorial_samples
    
    # Generate equatorial samples (-45° to +45°)
    equatorial_latitude = (torch.rand(equatorial_samples) * 90 - 45).requires_grad_()
    
    # Generate remaining samples for polar regions
    polar_latitude_north = (torch.rand(polar_samples // 2) * 45 + 45).requires_grad_()
    polar_latitude_south = (torch.rand(polar_samples // 2) * -45 - 45).requires_grad_()
    
    # Combine latitudes
    latitude = torch.cat([equatorial_latitude, polar_latitude_north, polar_latitude_south])
    
    # Generate other parameters
    longitude = (torch.rand(n_samples) * 360 - 180).requires_grad_()
    time = (torch.rand(n_samples) * 24).requires_grad_()
    slope = (torch.rand(n_samples) * 45).requires_grad_()
    aspect = (torch.rand(n_samples) * 360).requires_grad_()
    
    # Add 30% more samples in equatorial region (-15° to +15°)
    extra_equatorial = int(n_samples * 0.3)
    extra_latitude = (torch.rand(extra_equatorial) * 30 - 15).requires_grad_()
    extra_longitude = (torch.rand(extra_equatorial) * 360 - 180).requires_grad_()
    extra_time = (torch.rand(extra_equatorial) * 24).requires_grad_()
    extra_slope = (torch.rand(extra_equatorial) * 45).requires_grad_()
    extra_aspect = (torch.rand(extra_equatorial) * 360).requires_grad_()
    
    # Combine with original data
    latitude = torch.cat([latitude[:n_samples-extra_equatorial], extra_latitude])
    longitude = torch.cat([longitude[:n_samples-extra_equatorial], extra_longitude])
    time = torch.cat([time[:n_samples-extra_equatorial], extra_time])
    slope = torch.cat([slope[:n_samples-extra_equatorial], extra_slope])
    aspect = torch.cat([aspect[:n_samples-extra_equatorial], extra_aspect])
    
    # Add more samples around sunrise and sunset times
    twilight_samples = int(n_samples * 0.2)  # 20% of samples around twilight periods
    twilight_times = torch.cat([
        torch.normal(6, 1, (twilight_samples//2,)),  # Around sunrise
        torch.normal(18, 1, (twilight_samples//2,))  # Around sunset
    ]).requires_grad_()
    time[-twilight_samples:] = torch.clamp(twilight_times, 0, 24)
    
    # Normalize inputs
    lat_norm = latitude / 90
    lon_norm = longitude / 180
    time_norm = time / 24
    slope_norm = slope / 180
    aspect_norm = aspect / 360
    
    # Create normalized input tensor
    x_data = torch.stack([lat_norm, lon_norm, time_norm, slope_norm, aspect_norm], dim=1).float()
    
    # Calculate theoretical clear-sky irradiance using physics validator
    physics_model = SolarPhysicsIdeal()
    y_data = []
    
    for i in range(n_samples):
        irradiance = physics_model.calculate_irradiance(
            latitude[i], time[i], slope[i], aspect[i]
        )
        # Normalize irradiance with enhanced night time handling
        irradiance_norm = irradiance / physics_model.solar_constant
        y_data.append(irradiance_norm)
    
    y_data = torch.stack(y_data).reshape(-1, 1).float().requires_grad_()
    
    # Split into train and validation sets
    n_val = int(n_samples * validation_split)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]
    
    return (x_data[train_indices], y_data[train_indices], 
            x_data[val_indices], y_data[val_indices])

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Import physics validator
    from physics_validator import SolarPhysicsIdeal, calculate_metrics
    
    # Create model and trainer
    model = SolarPINN()
    trainer = PINNTrainer(model)
    physics_model = SolarPhysicsIdeal()
    
    # Generate training and validation data
    x_train, y_train, x_val, y_val = generate_training_data()
    
    # Enhanced training parameters
    n_epochs = 800  # Increased epochs
    batch_size = 256  # Increased batch size for better gradient estimates
    n_batches = len(x_train) // batch_size
    best_val_loss = float('inf')
    patience = 50  # Increased patience
    patience_counter = 0
    min_lr = 1e-6  # Reduced minimum learning rate
    
    # Learning rate scheduler with warm-up and cosine annealing
    def get_lr(epoch):
        warmup_epochs = 100  # Increased warmup epochs
        if epoch < warmup_epochs:
            return 0.002 * (epoch + 1) / warmup_epochs
        else:
            return max(min_lr, 0.001 * 0.5 * (1 + np.cos((epoch - warmup_epochs) * np.pi / (n_epochs - warmup_epochs))))
    
    # Physics loss weight scheduling with gradual increase
    initial_physics_weight = 0.05  # Reduced initial physics weight
    max_physics_weight = 0.3  # Reduced maximum physics weight
    
    print("Starting training...")
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    for epoch in range(n_epochs):
        # Update learning rate
        current_lr = get_lr(epoch)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Training
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        indices = torch.randperm(len(x_train))
        
        for i in range(n_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Update physics weight gradually
            physics_weight = min(
                initial_physics_weight + (max_physics_weight - initial_physics_weight) * (epoch / n_epochs),
                max_physics_weight
            )
            
            # Training step with current physics weight
            loss = trainer.train_step(x_batch, y_batch, physics_weight=physics_weight)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            try:
                # PINN predictions
                y_pred = model(x_val)
                val_loss = F.mse_loss(y_pred, y_val)
                
                # Calculate validation metrics
                val_metrics = calculate_metrics(y_val, y_pred)
            except Exception as e:
                print(f"Validation error: {str(e)}")
                val_loss = float('inf')
                val_metrics = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics
            }, 'best_solar_pinn_ideal.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}, Physics Weight: {physics_weight:.3f}")
            print(f"Validation Metrics:")
            print(f"MAE: {val_metrics['mae']:.4f}")
            print(f"RMSE: {val_metrics['rmse']:.4f}")
            print(f"R²: {val_metrics['r2']:.4f}")
            
            # Print example predictions
            sample_idx = torch.randint(0, len(x_val), (5,))
            y_sample = model(x_val[sample_idx])
            y_true = y_val[sample_idx]
            print("\nSample Predictions vs True Values:")
            for i in range(5):
                print(f"Pred: {y_sample[i].item():.4f}, True: {y_true[i].item():.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
