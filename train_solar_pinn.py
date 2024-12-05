import torch
import numpy as np
from solar_pinn import SolarPINN

def generate_training_data(n_samples=1000):
    """Generate synthetic training data"""
    # Random sampling of input parameters
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    longitude = torch.rand(n_samples) * 360 - 180  # -180 to 180 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    day_of_year = torch.rand(n_samples) * 365  # 1 to 365
    slope = torch.zeros(n_samples)  # Assuming flat surface for simplicity
    panel_azimuth = torch.zeros(n_samples)  # Assuming south-facing
    
    return latitude, longitude, time, day_of_year, slope, panel_azimuth

def main():
    # Create model
    model = SolarPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Further reduced learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Set up gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Reduced max norm
    
    # Generate training data
    latitude, longitude, time, day_of_year, slope, panel_azimuth = generate_training_data()
    
    # Training loop parameters
    n_epochs = 500  # Increased epochs
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_irradiance = model(latitude, longitude, time, day_of_year, slope, panel_azimuth)
        
        # Generate synthetic targets
        declination = model.calculate_declination(day_of_year)
        target_irradiance = model.calculate_top_of_atmosphere_irradiance(
            latitude, declination, (time - 12) * 15
        ).reshape(-1, 1)
        
        # Compute loss
        loss = model.compute_loss(pred_irradiance, target_irradiance, latitude, time, day_of_year)
        
        # Backward pass and optimization
        loss.backward()
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
