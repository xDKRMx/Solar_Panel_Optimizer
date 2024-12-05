import torch
import numpy as np
from solar_pinn import SolarPINN

def generate_data(n_samples=1000):
    """Generate synthetic data for training or validation"""
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    longitude = torch.rand(n_samples) * 360 - 180  # -180 to 180 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    day_of_year = torch.rand(n_samples) * 365  # 1 to 365
    slope = torch.zeros(n_samples)  # Assuming flat surface
    panel_azimuth = torch.zeros(n_samples)  # Assuming south-facing
    
    return latitude, longitude, time, day_of_year, slope, panel_azimuth

def main():
    # Create model
    model = SolarPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Generate training data
    latitude, longitude, time, day_of_year, slope, panel_azimuth = generate_data()
    
    # Training loop parameters
    n_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred_irradiance = model(latitude, longitude, time, day_of_year, slope, panel_azimuth)
        
        # Generate ideal targets using physical calculations
        declination = model.calculate_declination(day_of_year)
        hour_angle = model.calculate_hour_angle(time)
        zenith_angle = model.calculate_zenith_angle(latitude, declination, hour_angle)
        
        # Calculate ideal clear-sky irradiance
        cos_zenith = torch.cos(zenith_angle)
        target_irradiance = model.solar_constant * torch.clamp(cos_zenith, 0, 1)
        target_irradiance = target_irradiance.reshape(-1, 1)
        
        # Compute loss
        loss = model.compute_loss(pred_irradiance, target_irradiance, latitude, time, day_of_year)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()