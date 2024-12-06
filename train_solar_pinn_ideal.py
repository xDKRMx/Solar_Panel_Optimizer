import torch
import numpy as np
from solar_pinn_ideal import SolarPINN, PINNTrainer

def generate_training_data(n_samples=1000, validation_split=0.2):
    """Generate synthetic training data for ideal clear sky conditions."""
    # Generate random input parameters
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    longitude = torch.rand(n_samples) * 360 - 180  # -180 to 180 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    slope = torch.rand(n_samples) * 45  # 0 to 45 degrees slope
    aspect = torch.rand(n_samples) * 360  # 0 to 360 degrees aspect
    
    # Create input tensor
    x_data = torch.stack([latitude, longitude, time, slope, aspect], dim=1).float()
    
    # Calculate theoretical clear-sky irradiance for targets
    model = SolarPINN()
    with torch.no_grad():
        y_data = model.calculate_max_possible_irradiance(
            latitude, time
        ) * model.calculate_atmospheric_transmission(
            latitude, time
        ) * model.calculate_surface_orientation_factor(
            latitude, longitude, time, slope, aspect
        )
        y_data = y_data.reshape(-1, 1).float()
    
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
    model = SolarPINN()
    trainer = PINNTrainer(model)
    
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
            
            loss = trainer.train_step(x_batch, y_batch)
            epoch_loss += loss
        
        avg_train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = trainer.train_step(x_val, y_val)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
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
            
            # Print example predictions
            with torch.no_grad():
                sample_idx = torch.randint(0, len(x_val), (5,))
                y_sample = model(x_val[sample_idx])
                y_true = y_val[sample_idx]
                print("\nSample Predictions vs True Values:")
                for i in range(5):
                    print(f"Pred: {y_sample[i].item():.2f} W/m², True: {y_true[i].item():.2f} W/m²")
    
    print("Training completed!")

if __name__ == "__main__":
    main()