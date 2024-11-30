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

def validate_parameters(latitude, longitude, hour, day_number, slope, aspect, atm_transmission):
    """Validate input parameters against physical constraints"""
    warnings = []
    
    # Calculate daylight hours
    daylight_hours = processor.get_daylight_hours(latitude, day_number)
    sunrise = 12 - daylight_hours/2
    sunset = 12 + daylight_hours/2
    
    if hour < sunrise or hour > sunset:
        warnings.append(f"Warning: Selected time ({hour:.1f}) is outside daylight hours ({sunrise:.1f} to {sunset:.1f})")
    
    if abs(latitude) > 35:
        warnings.append("Warning: Location is outside optimal solar belt (-35° to 35° latitude)")
    
    optimal_slope = processor.get_optimal_tilt(latitude)
    if abs(slope - optimal_slope) > 20:
        warnings.append(f"Warning: Panel slope deviates significantly from optimal ({optimal_slope:.1f}°)")
    
    optimal_aspect = processor.get_optimal_aspect(latitude)
    aspect_diff = min((aspect - optimal_aspect) % 360, (optimal_aspect - aspect) % 360)
    if aspect_diff > 45:
        warnings.append(f"Warning: Panel aspect deviates significantly from optimal ({optimal_aspect}°)")
    
    return warnings

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
    
    # Calculate daylight hours for current latitude and day
    daylight_hours = processor.get_daylight_hours(latitude, day_number)
    sunrise = 12 - daylight_hours/2
    sunset = 12 + daylight_hours/2
    
    # Constrain hour selection to daylight hours
    hour = st.sidebar.slider(
        "Hour of Day",
        float(max(0, sunrise)),
        float(min(24, sunset)),
        12.0,
        help="Limited to daylight hours for the selected location and date"
    )
    
    # Installation parameters with optimal value hints
    optimal_slope = processor.get_optimal_tilt(latitude)
    slope = st.sidebar.slider(
        f"Panel Slope (Optimal: {optimal_slope:.1f}°)",
        0.0, 90.0, optimal_slope
    )
    
    optimal_aspect = processor.get_optimal_aspect(latitude)
    aspect = st.sidebar.slider(
        f"Panel Aspect (Optimal: {optimal_aspect}°)",
        0.0, 360.0, float(optimal_aspect)
    )
    
    # Atmospheric parameters with realistic bounds
    atm_transmission = st.sidebar.slider(
        "Atmospheric Transmission",
        0.6, 0.8, 0.7,
        help="Typical values: 0.6 (hazy) to 0.8 (clear sky)"
    )
    
    # Validate parameters and display warnings
    warnings = validate_parameters(latitude, longitude, hour, day_number, slope, aspect, atm_transmission)
    for warning in warnings:
        st.warning(warning)
    
    # Create input data
    input_data = processor.prepare_data(
        np.array([latitude]),
        np.array([longitude]),
        np.array([hour]),
        np.array([slope]),
        np.array([aspect]),
        np.array([atm_transmission])
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
