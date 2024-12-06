import torch
import numpy as np
import torch.nn.functional as F
from solar_pinn_ideal import SolarPINN, PINNTrainer
from physics_validator import SolarPhysicsIdeal

def generate_training_data(n_samples=1000, validation_split=0.2):
    """Generate synthetic training data for ideal clear sky conditions."""
    # Generate random input parameters with normalization and better distribution
    latitude = (torch.rand(n_samples) * 180 - 90).requires_grad_()  # -90 to 90 degrees
    longitude = (torch.rand(n_samples) * 360 - 180).requires_grad_()  # -180 to 180 degrees
    time = (torch.rand(n_samples) * 24).requires_grad_()  # 0 to 24 hours
    slope = (torch.rand(n_samples) * 45).requires_grad_()  # 0 to 45 degrees slope
    aspect = (torch.rand(n_samples) * 360).requires_grad_()  # 0 to 360 degrees aspect
    
    # Ensure inputs are on CPU and have correct dtype
    latitude = latitude.cpu().float()
    longitude = longitude.cpu().float()
    time = time.cpu().float()
    slope = slope.cpu().float()
    aspect = aspect.cpu().float()
    
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
    batch_size = 32  # Restored to original batch size
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
    
    # Fixed physics weight
    physics_weight = 0.1  # Reset to original value
    
    print("Starting training...")
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    for epoch in range(n_epochs):
        try:
            # Training
            model.train()
            epoch_loss = 0
            
            # Shuffle training data
            indices = torch.randperm(len(x_train))
            
            for i in range(n_batches):
                try:
                    batch_indices = indices[i*batch_size:(i+1)*batch_size]
                    x_batch = x_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # Training step with fixed physics weight
                    loss = trainer.train_step(x_batch, y_batch, physics_weight=physics_weight)
                    
                    # Check for NaN loss
                    if torch.isnan(torch.tensor(loss)):
                        print(f"NaN loss detected in batch {i}, skipping...")
                        continue
                        
                    epoch_loss += loss
                except Exception as e:
                    print(f"Error in batch {i}: {str(e)}")
                    continue
            
        avg_train_loss = epoch_loss / n_batches
            
            # Step the learning rate scheduler
            scheduler.step()
        
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
        
        if (epoch + 1) % 5 == 0:
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