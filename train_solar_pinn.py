import torch
import numpy as np
from solar_pinn import SolarPINN, PINNTrainer

def generate_training_data(n_samples=1000):
    """Generate synthetic training data for ideal clear sky conditions."""
    # Random sampling of essential input parameters
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    day_of_year = torch.rand(n_samples) * 365 + 1  # 1 to 365
    slope = torch.zeros(n_samples)  # Flat surface for initial training
    
    return latitude, time, day_of_year, slope

def main():
    # Create model and trainer
    model = SolarPINN()
    trainer = PINNTrainer(model)
    
    # Generate training data
    latitude, time, day_of_year, slope = generate_training_data()
    
    # Training loop parameters
    n_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(n_epochs):
        # Calculate theoretical clear-sky irradiance for training targets
        declination = model.calculate_declination(day_of_year)
        hour_angle = model.calculate_hour_angle(time)
        zenith_angle = model.calculate_zenith_angle(latitude, declination, hour_angle)
        
        # Create target values (simplified clear sky model)
        cos_zenith = torch.cos(zenith_angle)
        target_irradiance = model.solar_constant * torch.clamp(cos_zenith, 0, 1)
        target_irradiance = target_irradiance.reshape(-1, 1)
        
        # Create input tensor
        x_data = torch.stack([latitude, time, day_of_year, slope], dim=1)
        
        # Training step
        x_data = torch.stack([latitude, time, day_of_year, slope], dim=1).float()
        target_irradiance = target_irradiance.float()
        loss = trainer.train_step(x_data, target_irradiance)
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_solar_pinn.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss:.4f}, Best Loss: {best_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
