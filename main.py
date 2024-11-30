import streamlit as st
import numpy as np
import torch
from models.pinn_model import SolarPINN, PINNTrainer
from utils.solar_physics import SolarIrradianceCalculator
from utils.data_processor import DataProcessor
from visualization.plots import SolarPlotter

# Initialize components
model = SolarPINN()
trainer = PINNTrainer(model)
calculator = SolarIrradianceCalculator()
processor = DataProcessor()
plotter = SolarPlotter()

def main():
    st.title("Solar Panel Placement Optimizer")
    st.write("Physics-Informed Neural Network for Optimal Solar Panel Placement")
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    
    # Location parameters
    latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 0.0)
    longitude = st.sidebar.slider("Longitude", -180.0, 180.0, 0.0)
    
    # Time parameters
    day_number = st.sidebar.slider("Day of Year", 1, 365, 182)
    hour = st.sidebar.slider("Hour of Day", 0, 24, 12)
    
    # Installation parameters
    slope = st.sidebar.slider("Panel Slope", 0.0, 90.0, 30.0)
    aspect = st.sidebar.slider("Panel Aspect", 0.0, 360.0, 180.0)
    
    # Atmospheric parameters
    st.sidebar.subheader("Atmospheric Parameters")
    atm_transmission = st.sidebar.slider("Atmospheric Transmission", 0.0, 1.0, 0.7)
    cloud_cover = st.sidebar.slider("Cloud Cover", 0.0, 1.0, 0.0)
    wavelength = st.sidebar.slider("Wavelength (μm)", 0.3, 1.0, 0.5)
    
    # Create input data
    input_data = processor.prepare_data(
        np.array([latitude]),
        np.array([longitude]),
        np.array([hour]),
        np.array([slope]),
        np.array([aspect]),
        np.array([atm_transmission]),
        np.array([cloud_cover]),
        np.array([wavelength])
    )
    
    # Model prediction
    with torch.no_grad():
        prediction = model(input_data)
    
    # Calculate physics-based irradiance
    physics_irradiance = calculator.calculate_irradiance(
        latitude, longitude, day_number, hour,
        slope, aspect, atm_transmission
    )
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PINN Prediction")
        st.write(f"Predicted Irradiance: {prediction.item():.2f} W/m²")
    
    with col2:
        st.subheader("Physics Calculation")
        st.write(f"Physics-based Irradiance: {physics_irradiance:.2f} W/m²")
    
    # Optimization analysis
    if st.button("Run Optimization Analysis"):
        slopes = np.linspace(0, 90, 30)
        aspects = np.linspace(0, 360, 30)
        
        efficiency = np.zeros((30, 30))
        for i, s in enumerate(slopes):
            for j, a in enumerate(aspects):
                input_data = processor.prepare_data(
                    np.array([latitude]),
                    np.array([longitude]),
                    np.array([hour]),
                    np.array([s]),
                    np.array([a]),
                    np.array([atm_transmission])
                )
                with torch.no_grad():
                    efficiency[i, j] = model(input_data).item()
        
        # Plot optimization results
        fig = plotter.plot_optimization_results(slopes, aspects, efficiency)
        st.plotly_chart(fig)
        
        # Find optimal parameters
        opt_idx = np.unravel_index(np.argmax(efficiency), efficiency.shape)
        st.success(f"""
        Optimal Parameters:
        - Slope: {slopes[opt_idx[0]]:.1f}°
        - Aspect: {aspects[opt_idx[1]]:.1f}°
        - Expected Efficiency: {efficiency[opt_idx]:.2f}
        """)

if __name__ == "__main__":
    main()
