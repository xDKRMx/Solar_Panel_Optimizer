import torch
import numpy as np
import torch.nn.functional as F
from solar_pinn_ideal import SolarPINN, PINNTrainer
from physics_validator import SolarPhysicsIdeal

def generate_training_data(n_samples=1000, validation_split=0.2):
    """Generate synthetic training data for ideal clear sky conditions with balanced hemisphere sampling."""
    # Define latitude ranges for stratified sampling
    n_south = int(n_samples * 0.5)  # Ensure 50% of samples are from southern hemisphere
    n_deep_south = int(n_south * 0.4)  # 40% of southern samples from -90° to -45°
    n_mid_south = n_south - n_deep_south  # Remaining southern samples from -45° to 0°
    n_north = n_samples - n_south  # Northern hemisphere samples
    
    # Generate stratified latitude samples
    deep_south = (torch.rand(n_deep_south) * 45 - 90).requires_grad_()  # -90° to -45°
    mid_south = (torch.rand(n_mid_south) * 45 - 45).requires_grad_()  # -45° to 0°
    north = (torch.rand(n_north) * 90).requires_grad_()  # 0° to 90°
    
    # Combine latitude samples
    latitude = torch.cat([deep_south, mid_south, north])
    
    # Generate other parameters
    longitude = (torch.rand(n_samples) * 360 - 180).requires_grad_()  # -180° to 180°
    
    # Generate seasonal variations
    seasons = torch.linspace(0, 365, 4)  # Four seasons
    time_offsets = torch.randn(n_samples) * 2 + 12  # Peak around noon with variation
    time = torch.remainder(time_offsets, 24).requires_grad_()  # Ensure time is within 0-24
    
    slope = (torch.rand(n_samples) * 45).requires_grad_()  # 0° to 45° slope
    aspect = (torch.rand(n_samples) * 360).requires_grad_()  # 0° to 360° aspect
    
    # Enhanced normalization for better hemisphere representation
    lat_norm = torch.where(
        latitude < 0,
        latitude / 90 * 1.2,  # Increase weight for southern hemisphere
        latitude / 90
    )
    
    # Calculate seasonal weights
    day_of_year = torch.floor(time / 24 * 365)
    seasonal_weight = torch.abs(torch.sin(2 * torch.pi * day_of_year / 365))
    
    # Apply seasonal corrections to normalization
    lon_norm = longitude / 180
    time_norm = time / 24
    slope_norm = slope / 180
    aspect_norm = aspect / 360
    
    # Create normalized input tensor with seasonal information
    x_data = torch.stack([
        lat_norm,
        lon_norm,
        time_norm,
        slope_norm,
        aspect_norm,
        seasonal_weight  # Add seasonal weight as additional feature
    ], dim=1).float()
    
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
    
    # Training parameters
    n_epochs = 200
    batch_size = 64  # Increased batch size
    n_batches = len(x_train) // batch_size
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, 
        step_size=50,
        gamma=0.5
    )
    
    # Dynamic physics loss weight scheduling based on hemisphere
    initial_physics_weight = 0.15
    max_physics_weight = 0.3
    physics_weight = initial_physics_weight
    
    # Separate tracking for hemisphere performance
    south_metrics = {'loss': float('inf'), 'samples': 0}
    north_metrics = {'loss': float('inf'), 'samples': 0}
    
    # Enhanced validation sets for hemisphere-specific monitoring
    def get_hemisphere_metrics(y_true, y_pred, latitude):
        is_south = latitude < 0
        south_mask = is_south
        north_mask = ~is_south
        
        south_loss = F.mse_loss(y_pred[south_mask], y_true[south_mask]) if south_mask.any() else torch.tensor(0.0)
        north_loss = F.mse_loss(y_pred[north_mask], y_true[north_mask]) if north_mask.any() else torch.tensor(0.0)
        
        return {
            'south': south_loss.item(),
            'north': north_loss.item(),
            'south_samples': south_mask.sum().item(),
            'north_samples': north_mask.sum().item()
        }
    
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
            
            # Calculate hemisphere-specific weights
            batch_latitudes = x_batch[:, 0] * 90  # Denormalize latitude
            is_south = batch_latitudes < 0
            
            # Adjust physics weight based on hemisphere performance
            south_weight = physics_weight * 1.2 if south_metrics['loss'] > north_metrics['loss'] else physics_weight
            north_weight = physics_weight
            
            # Apply hemisphere-specific physics weights
            batch_weights = torch.where(
                is_south,
                torch.tensor(south_weight, dtype=torch.float32),
                torch.tensor(north_weight, dtype=torch.float32)
            )
            
            # Training step with dynamic weights
            loss = trainer.train_step(x_batch, y_batch, physics_weight=batch_weights.to(x_batch.device))
            epoch_loss += loss
            
            # Update hemisphere-specific metrics
            hemisphere_metrics = get_hemisphere_metrics(y_batch, model(x_batch), batch_latitudes)
            south_metrics['loss'] = hemisphere_metrics['south']
            south_metrics['samples'] = hemisphere_metrics['south_samples']
            north_metrics['loss'] = hemisphere_metrics['north']
            north_metrics['samples'] = hemisphere_metrics['north_samples']
            
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