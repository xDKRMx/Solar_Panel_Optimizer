import torch
import matplotlib.pyplot as plt
from solar_pinn_ideal import SolarPINN
import numpy as np

def load_and_visualize_model(model_path='best_solar_pinn_ideal.pth'):
    # Create model instance
    model = SolarPINN()
    
    # Load trained weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test data for visualization
    latitudes = torch.linspace(-90, 90, 180)
    times = torch.linspace(0, 24, 24)
    day_of_year = torch.tensor([172])  # Summer solstice
    slope = torch.zeros_like(latitudes)
    aspect = torch.zeros_like(latitudes)
    
    # Create meshgrid for latitude and time
    lat_grid, time_grid = torch.meshgrid(latitudes, times)
    
    # Prepare input data
    lat_norm = lat_grid.flatten() / 90
    time_norm = time_grid.flatten() / 24
    slope_norm = torch.zeros_like(lat_norm)
    aspect_norm = torch.zeros_like(lat_norm)
    lon_norm = torch.zeros_like(lat_norm)
    
    x_data = torch.stack([lat_norm, lon_norm, time_norm, slope_norm, aspect_norm], dim=1)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(x_data)
    
    # Reshape predictions for plotting
    irradiance_map = predictions.reshape(len(times), len(latitudes))
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(irradiance_map, 
               extent=[-90, 90, 0, 24], 
               aspect='auto', 
               origin='lower',
               cmap='viridis')
    plt.colorbar(label='Normalized Solar Irradiance')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Time of Day (hours)')
    plt.title('Predicted Solar Irradiance Distribution')
    plt.savefig('solar_irradiance_prediction.png')
    plt.close()

if __name__ == "__main__":
    load_and_visualize_model()
