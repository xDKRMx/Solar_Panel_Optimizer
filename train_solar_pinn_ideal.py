import torch
import numpy as np
import torch.nn.functional as F
from solar_pinn_ideal import SolarPINN, PINNTrainer
from physics_validator import SolarPhysicsIdeal

def generate_training_data(n_samples=3000, validation_split=0.2):
    """Generate synthetic training data for ideal clear sky conditions with enhanced sampling."""
    # Increase focused samples for problematic ranges
    n_focused = int(n_samples * 0.8)  # 80% of samples in target range
    n_regular = n_samples - n_focused
    
    # Add additional samples around sunrise/sunset periods
    n_transition = int(n_focused * 0.3)  # 30% of focused samples for transition periods
    
    # Implement stratified sampling for day_of_year
    seasons = [(1, 91), (92, 182), (183, 273), (274, 365)]  # Spring, Summer, Fall, Winter
    samples_per_season = n_samples // 4
    
    # Generate focused samples for -90° to +60° range
    lat_focused = (torch.rand(n_focused) * 150 - 90).requires_grad_()  # -90 to +60 degrees
    
    # Generate regular samples for remaining range
    lat_regular = (torch.rand(n_regular) * 30 + 60).requires_grad_()  # +60 to +90 degrees
    
    # Combine latitude samples
    latitude = torch.cat([lat_focused, lat_regular])
    
    # Generate other parameters
    longitude = (torch.rand(n_samples) * 360 - 180).requires_grad_()  # -180 to 180 degrees
    time = (torch.rand(n_samples) * 24).requires_grad_()  # 0 to 24 hours (just hours now)
    day_of_year = (torch.floor(torch.rand(n_samples) * 365) + 1).requires_grad_()  # 1 to 365
    slope = (torch.rand(n_samples) * 45).requires_grad_()  # 0 to 45 degrees slope
    aspect = (torch.rand(n_samples) * 360).requires_grad_()  # 0 to 360 degrees aspect
    
    # Normalize inputs
    lat_norm = latitude / 90
    lon_norm = longitude / 180
    time_norm = time / 24
    day_norm = (day_of_year - 1) / 364  # Normalize to [0,1]
    slope_norm = slope / 180
    aspect_norm = aspect / 360
    
    # Create normalized input tensor
    x_data = torch.stack([lat_norm, lon_norm, time_norm, day_norm, slope_norm, aspect_norm], dim=1).float()
    
    # Calculate theoretical clear-sky irradiance using physics validator
    physics_model = SolarPhysicsIdeal()
    y_data = []
    
    for i in range(n_samples):
        # Use original values for physics calculation
        irradiance = physics_model.calculate_irradiance(
            latitude[i], time[i], slope[i], aspect[i]
        )
        # Normalize irradiance
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
    
    # Enhanced training parameters with optimizations
    n_epochs = 200
    batch_size = 512  # Increased batch size for better stability
    n_batches = len(x_train) // batch_size
    best_val_loss = float('inf')
    min_delta = 1e-5  # Updated minimum improvement threshold
    patience = 20  # Reduced patience
    patience_counter = 0
    
    # Learning rate scheduling with warmup
    initial_lr = 0.001
    warmup_epochs = 5
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Use ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Enhanced physics loss weight scheduling with cosine annealing
    initial_physics_weight = 0.05
    max_physics_weight = 0.3
    physics_weight = initial_physics_weight
    
    print("Starting training...")
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        indices = torch.randperm(len(x_train))
        
        for i in range(n_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Update physics weight using cosine annealing
            physics_weight = initial_physics_weight + 0.5 * (max_physics_weight - initial_physics_weight) * \
                           (1 + np.cos(np.pi * epoch / n_epochs))
            
            # Training step with current physics weight and gradient clipping
            loss = trainer.train_step(x_batch, y_batch, physics_weight=physics_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
            epoch_loss += loss
            
        # Step the learning rate scheduler
        scheduler.step()
        
        avg_train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            try:
                # PINN predictions with gradient tracking disabled
                y_pred = model(x_val)
                val_loss = F.mse_loss(y_pred, y_val)
                
                # Calculate validation metrics
                val_metrics = calculate_metrics(y_val, y_pred)
            except Exception as e:
                print(f"Validation error: {str(e)}")
                val_loss = float('inf')
                val_metrics = {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        # Early stopping check
        # Early stopping with minimum delta threshold
        if val_loss < best_val_loss - min_delta:
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
        
        if (epoch + 1) % 5 == 0:  # Print more frequently
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Validation Metrics:")
            print(f"MAE: {val_metrics['mae']:.2f} W/m²")
            print(f"RMSE: {val_metrics['rmse']:.2f} W/m²")
            print(f"R²: {val_metrics['r2']:.4f}")
            
            # Print example predictions
            sample_idx = torch.randint(0, len(x_val), (5,))
            y_sample = model(x_val[sample_idx])
            y_true = y_val[sample_idx]
            print("\nSample Predictions vs True Values:")
            for i in range(5):
                print(f"Pred: {y_sample[i].item():.2f} W/m², True: {y_true[i].item():.2f} W/m²")
    
    print("Training completed!")

if __name__ == "__main__":
    main()