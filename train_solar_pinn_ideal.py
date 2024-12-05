import torch
import numpy as np
from solar_pinn_ideal import SolarPINNIdeal, SolarPINNTrainer

def generate_training_data(n_samples=1000, validation_split=0.2):
    """Generate synthetic training data for ideal clear sky conditions."""
    # Generate random input parameters
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    day_of_year = torch.rand(n_samples) * 365 + 1  # 1 to 365
    slope = torch.rand(n_samples) * 45  # 0 to 45 degrees slope
    
    # Create input tensor
    x_data = torch.stack([latitude, time, day_of_year, slope], dim=1).float()
    
    # Calculate theoretical clear-sky irradiance for targets
    model = SolarPINNIdeal()
    with torch.no_grad():
        y_data = model.calculate_clear_sky_irradiance(
            model.calculate_zenith_angle(
                latitude,
                model.calculate_declination(day_of_year),
                model.calculate_hour_angle(time)
            ),
            model.calculate_air_mass(
                model.calculate_zenith_angle(
                    latitude,
                    model.calculate_declination(day_of_year),
                    model.calculate_hour_angle(time)
                )
            ),
            model.calculate_surface_factor(
                model.calculate_zenith_angle(
                    latitude,
                    model.calculate_declination(day_of_year),
                    model.calculate_hour_angle(time)
                ),
                slope
            )
        ).reshape(-1, 1).float()
    
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
    
    # Create model and trainer
    model = SolarPINNIdeal()
    trainer = SolarPINNTrainer(model)
    
    # Generate training and validation data
    x_train, y_train, x_val, y_val = generate_training_data()
    
    # Training parameters
    n_epochs = 200
    batch_size = 32
    n_batches = len(x_train) // batch_size
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
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
            
            loss, metrics = trainer.train_step(x_batch, y_batch)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        val_loss, val_metrics = trainer.train_step(x_val, y_val, validation=True)
        
        # Store metrics
        trainer.metrics['train_loss'].append(avg_train_loss)
        trainer.metrics['val_loss'].append(val_loss)
        trainer.metrics['rmse'].append(val_metrics['rmse'])
        trainer.metrics['mae'].append(val_metrics['mae'])
        trainer.metrics['r2'].append(val_metrics['r2'])
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': trainer.metrics,
                'epoch': epoch
            }, 'best_solar_pinn_ideal.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:  # Print more frequently
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"RMSE: {val_metrics['rmse']:.4f} W/m²")
            print(f"MAE: {val_metrics['mae']:.4f} W/m²")
            print(f"R²: {val_metrics['r2']:.4f}")
            
            # Print example predictions
            with torch.no_grad():
                sample_idx = torch.randint(0, len(x_val), (5,))
                y_sample = model(x_val[sample_idx])
                y_true = y_val[sample_idx]
                print("\nSample Predictions vs True Values:")
                for i in range(5):
                    print(f"Pred: {y_sample[i].item():.2f} W/m², True: {y_true[i].item():.2f} W/m²")
    
    print("Training completed!")
    print("Final validation metrics:")
    print(f"RMSE: {trainer.metrics['rmse'][-1]:.4f}")
    print(f"MAE: {trainer.metrics['mae'][-1]:.4f}")
    print(f"R²: {trainer.metrics['r2'][-1]:.4f}")

if __name__ == "__main__":
    main()
