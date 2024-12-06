import plotly.graph_objects as go
import numpy as np

class SolarPlotter:
    def __init__(self):
        self.colorscale = 'Viridis'
        
    def plot_irradiance_heatmap(self, latitudes, longitudes, irradiance):
        """Create heatmap of solar irradiance"""
        fig = go.Figure(data=go.Heatmap(
            z=irradiance,
            x=longitudes,
            y=latitudes,
            colorscale=self.colorscale,
            colorbar=dict(title='Solar Irradiance (W/m²)')
        ))
        
        fig.update_layout(
            title='Solar Irradiance Distribution',
            xaxis_title='Longitude',
            yaxis_title='Latitude'
        )
        
        return fig
    
    def plot_optimization_results(self, slopes, aspects, efficiency):
        """Create 3D surface plot of optimization results"""
        fig = go.Figure(data=go.Surface(
            x=slopes,
            y=aspects,
            z=efficiency,
            colorscale=self.colorscale
        ))
        
        fig.update_layout(
            title='Solar Panel Efficiency by Orientation',
            scene=dict(
                xaxis_title='Slope (degrees)',
                yaxis_title='Aspect (degrees)',
                zaxis_title='Efficiency'
            )
        )
        
        return fig
        
    def create_world_irradiance_map(self, model, hour, day_of_year, resolution=30):
        """Create an interactive world map showing solar irradiance predictions"""
        import torch
        import numpy as np
        
        # Create latitude and longitude grid
        lats = np.linspace(-90, 90, resolution)
        lons = np.linspace(-180, 180, resolution)
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        
        # Initialize irradiance grid
        irradiance = np.zeros((resolution, resolution))
        
        # Calculate irradiance for each point
        with torch.no_grad():
            for i in range(resolution):
                for j in range(resolution):
                    # Prepare input tensor
                    x_data = torch.tensor([[
                        lat_grid[i, j] / 90,  # Normalize latitude
                        lon_grid[i, j] / 180,  # Normalize longitude
                        hour / 24,  # Normalize hour
                        0 / 180,  # Default slope (horizontal)
                        180 / 360  # Default aspect (south-facing)
                    ]], dtype=torch.float32)
                    
                    # Get prediction
                    prediction = model(x_data)
                    irradiance[i, j] = prediction.item() * model.solar_constant
        
        # Create choropleth map
        fig = go.Figure()
        
        # Add contour map of irradiance
        fig.add_trace(go.Contourcarpet(
            a=lon_grid.flatten(),
            b=lat_grid.flatten(),
            z=irradiance.flatten(),
            colorscale=self.colorscale,
            colorbar=dict(
                title='Solar Irradiance (W/m²)',
                titleside='right'
            ),
        ))
        
        # Add world map base layer
        fig.add_trace(go.Scattergeo(
            lon=[],
            lat=[],
            mode='lines',
            line=dict(width=1, color='gray'),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Global Solar Irradiance Distribution',
            geo=dict(
                showland=True,
                showcountries=True,
                showocean=True,
                countrywidth=0.5,
                landcolor='rgb(243, 243, 243)',
                oceancolor='rgb(204, 229, 255)',
                projection_type='equirectangular',
                showcoastlines=True,
                coastlinewidth=1,
                coastlinecolor='rgb(80, 80, 80)'
            ),
            width=1000,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
