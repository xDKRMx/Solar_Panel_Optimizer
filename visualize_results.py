import torch
import plotly.graph_objects as go
from solar_pinn_ideal import SolarPINN

def load_model(model_path='best_solar_pinn_ideal.pth'):
    """Load the trained PINN model."""
    try:
        model = SolarPINN()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def create_surface_plot(model, latitude, longitude, time, resolution=30):
    """Create interactive 3D surface plot of solar panel efficiency using Plotly."""
    try:
        from physics_validator import SolarPhysicsIdeal
        physics_model = SolarPhysicsIdeal()
        
        # Convert inputs to tensors with correct dtype
        latitude = torch.tensor(float(latitude), dtype=torch.float32)
        longitude = torch.tensor(float(longitude), dtype=torch.float32)
        time = torch.tensor(float(time), dtype=torch.float32)
        
        # Create meshgrid for slope and aspect
        slopes = torch.linspace(0, 90, resolution)  # Panel slope (β) from 0 to 90 degrees
        aspects = torch.linspace(0, 360, resolution)  # Panel azimuth (φp) from 0 to 360 degrees
        slope_grid, aspect_grid = torch.meshgrid(slopes, aspects, indexing='xy')
        
        # Initialize results tensor
        efficiency_map = torch.zeros((resolution, resolution), dtype=torch.float32)
        
        # Default atmospheric conditions
        atm_condition = torch.ones(1, dtype=torch.float32) * 0.8  # Clear sky condition
        cloud_cover = torch.zeros(1, dtype=torch.float32)  # No clouds
        wavelength = torch.ones(1, dtype=torch.float32) * 0.5  # Default wavelength (μm)
        
        # Calculate efficiency for each point using PINN model
        for i in range(resolution):
            for j in range(resolution):
                # Prepare normalized input tensor for PINN
                input_tensor = torch.stack([
                    latitude / 90.0,  # Normalize latitude to [-1, 1]
                    longitude / 180.0,  # Normalize longitude to [-1, 1]
                    time / 24.0,  # Normalize time to [0, 1]
                    slopes[i] / 180.0,  # Normalize slope to [0, 1]
                    aspects[j] / 360.0,  # Normalize aspect to [0, 1]
                    atm_condition,  # Already normalized
                    cloud_cover,  # Already normalized
                    wavelength  # Already normalized
                ]).reshape(1, -1)
                
                # Get irradiance prediction from PINN model
                with torch.no_grad():
                    predicted_irradiance = model(input_tensor) * physics_model.solar_constant
                
                # Calculate efficiency using the predicted irradiance
                efficiency = physics_model.calculate_efficiency(
                    latitude=latitude,
                    time=time,
                    predicted_irradiance=predicted_irradiance.item(),
                    slope=slopes[i],
                    panel_azimuth=aspects[j]
                )
                efficiency_map[i, j] = efficiency.item()
        
        # Find optimal parameters
        max_idx = torch.argmax(efficiency_map)
        optimal_slope = slopes[max_idx // resolution].item()
        optimal_aspect = aspects[max_idx % resolution].item()
        max_efficiency = efficiency_map.max().item()
        
        # Create Plotly figure
        fig = go.Figure(data=[
            go.Surface(
                x=slope_grid.numpy(),
                y=aspect_grid.numpy(),
                z=efficiency_map.numpy(),
                colorscale='viridis',
                colorbar=dict(title='Efficiency'),
                hovertemplate=(
                    "Slope: %{x:.1f}°<br>"
                    "Aspect: %{y:.1f}°<br>"
                    "Efficiency: %{z:.3f}<br>"
                    "<extra></extra>"
                )
            )
        ])
        
        # Optimal point marker removed as requested
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Solar Panel Efficiency by Orientation<br>Latitude: {latitude:.1f}°, Time: {time:.1f}h',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='Panel Slope β (degrees)',
                yaxis_title='Panel Azimuth φp (degrees)',
                zaxis_title='Total Efficiency η',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis=dict(range=[0, 90]),   # Slope range
                yaxis=dict(range=[0, 360]),  # Azimuth range
                zaxis=dict(
                    range=[0, 0.25],  # Efficiency range (0-0.25)
                    tickformat='.3f'   # Format as decimal number with 3 decimal places
                )
            ),
            template="plotly_dark"
        )
        
        metrics = {
            'optimal_slope': optimal_slope,
            'optimal_aspect': optimal_aspect,
            'max_efficiency': max_efficiency
        }
        
        return fig, metrics
        
    except Exception as e:
        raise RuntimeError(f"Error in visualization: {str(e)}")

if __name__ == "__main__":
    model = load_model()
    fig, metrics = create_surface_plot(model, 45.0, 0.0, 12.0)
    
    print("\nOptimal Parameters:")
    print(f"Slope: {metrics['optimal_slope']:.2f}°")
    print(f"Aspect: {metrics['optimal_aspect']:.2f}°")
    print(f"Maximum Efficiency: {metrics['max_efficiency']:.3f}")
    
    fig.write_html('solar_panel_efficiency_3d.html')
