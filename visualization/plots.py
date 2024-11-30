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
