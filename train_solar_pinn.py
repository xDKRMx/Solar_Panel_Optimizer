import torch
import numpy as np
from solar_pinn import SolarPINN

def generate_training_data(n_samples=1000):
    """Generate synthetic training data for ideal clear sky conditions"""
    # Random sampling of essential input parameters
    latitude = torch.rand(n_samples) * 180 - 90  # -90 to 90 degrees
    time = torch.rand(n_samples) * 24  # 0 to 24 hours
    day_of_year = torch.rand(n_samples) * 365 + 1  # 1 to 365
    slope = torch.zeros(n_samples)  # Flat surface for initial training
    
    return latitude, time, day_of_year, slope

def main():
    # Create model
    model = SolarPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Generate training data
    latitude, time, day_of_year, slope = generate_training_data()
    
    # Training loop parameters
    n_epochs = 200
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with physics constraints
        pred_irradiance = model(latitude, time, day_of_year, slope)
        
        # Calculate theoretical clear-sky irradiance for training
        declination = model.calculate_declination(day_of_year)
        hour_angle = model.calculate_hour_angle(time)
        target_irradiance = model.calculate_toa_irradiance(latitude, declination, hour_angle)
        target_irradiance = target_irradiance.reshape(-1, 1)
        
        # Compute physics-informed loss
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
    
    # Save the trained model
    torch.save(model.state_dict(), 'solar_pinn_model.pth')

if __name__ == "__main__":
    main()
