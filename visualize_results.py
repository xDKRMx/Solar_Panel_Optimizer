import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solar_pinn_ideal import SolarPINN

def load_model(model_path='best_solar_pinn_ideal.pth'):
    """Load the trained PINN model."""
    model = SolarPINN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def create_surface_plot(model, latitude, longitude, time, resolution=30):
    """Create 3D surface plot of solar panel efficiency."""
    try:
        # Convert inputs to tensors if they aren't already
        latitude = torch.tensor(latitude, dtype=torch.float32)
        longitude = torch.tensor(longitude, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)
        
        # Create meshgrid for slope and aspect
        slopes = torch.linspace(0, 90, resolution)
        aspects = torch.linspace(0, 360, resolution)
        slope_grid, aspect_grid = torch.meshgrid(slopes, aspects, indexing='xy')
        
        # Prepare input data
        lat_norm = torch.full((resolution * resolution,), latitude / 90, dtype=torch.float32)
        lon_norm = torch.full((resolution * resolution,), longitude / 180, dtype=torch.float32)
        time_norm = torch.full((resolution * resolution,), time / 24, dtype=torch.float32)
        slope_norm = slope_grid.flatten() / 180
        aspect_norm = aspect_grid.flatten() / 360
        
        x_data = torch.stack([lat_norm, lon_norm, time_norm, slope_norm, aspect_norm], dim=1).float()
        
        # Get predictions and calculate efficiency
        with torch.no_grad():
            predictions = model(x_data)
            
            # Convert irradiance to efficiency (normalize by solar constant)
            efficiency = predictions * model.solar_constant  # De-normalize
            efficiency = efficiency / model.solar_constant   # Calculate efficiency ratio
            
            # Reshape predictions for plotting
            efficiency_map = efficiency.reshape(resolution, resolution)
            
            # Find optimal point
            max_idx = torch.argmax(efficiency)
            optimal_slope = slopes[max_idx // resolution].item()
            optimal_aspect = aspects[max_idx % resolution].item()
            max_efficiency = efficiency[max_idx].item()
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot surface
            surf = ax.plot_surface(slope_grid.numpy(), aspect_grid.numpy(), efficiency_map.numpy(),
                                cmap='viridis', alpha=0.8)
            
            # Highlight optimal point
            ax.scatter([optimal_slope], [optimal_aspect], [max_efficiency],
                    color='red', s=100, label='Optimal Point')
            
            # Customize plot
            ax.set_xlabel('Panel Slope (degrees)')
            ax.set_ylabel('Panel Aspect (degrees)')
            ax.set_zlabel('Relative Efficiency')
            ax.set_title(f'Solar Panel Efficiency by Orientation\nLatitude: {latitude:.1f}°, Time: {time:.1f}h')
            
            # Add colorbar
            plt.colorbar(surf, ax=ax, label='Efficiency')
            
            # Add optimal point annotation
            plt.legend()
            
            metrics = {
                'optimal_slope': optimal_slope,
                'optimal_aspect': optimal_aspect,
                'max_efficiency': max_efficiency,
                'relative_efficiency': efficiency_map / max_efficiency
            }
            
            return fig, metrics
            
    except Exception as e:
        raise RuntimeError(f"Error in visualization: {str(e)}")

def save_visualization(latitude=45.0, longitude=0.0, time=12.0):
    """Generate and save visualization with given parameters."""
    model = load_model()
    fig, metrics = create_surface_plot(model, latitude, longitude, time)
    
    # Print metrics
    print("\nOptimal Parameters:")
    print(f"Slope: {metrics['optimal_slope']:.2f}°")
    print(f"Aspect: {metrics['optimal_aspect']:.2f}°")
    print(f"Maximum Efficiency: {metrics['max_efficiency']:.3f}")
    
    # Save plot
    plt.savefig('solar_panel_efficiency_3d.png')
    plt.close()
    
    return metrics

if __name__ == "__main__":
    save_visualization()
