import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from solar_pinn_ideal import SolarPINN
from physics_validator import SolarPhysicsIdeal
from visualize_results import create_surface_plot, load_model

def main():
    st.title("Solar Panel Placement Optimizer")
    st.write("Physics-Informed Neural Network for Optimal Solar Panel Placement")
    
    # Load the trained model
    try:
        model = load_model('best_solar_pinn_ideal.pth')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    
    # Location parameters
    latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 45.0, 0.1)
    longitude = st.sidebar.slider("Longitude", -180.0, 180.0, 0.0, 0.1)
    
    # Time parameters
    day_of_year = st.sidebar.slider("Day of Year", 1, 365, 182)
    hour = st.sidebar.slider("Hour of Day", 0, 24, 12, 0.1)
    
    # Create 3D visualization
    if st.button("Generate 3D Surface Plot"):
        with st.spinner("Generating visualization..."):
            try:
                fig, metrics = create_surface_plot(model, latitude, longitude, hour)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Optimal Slope", f"{metrics['optimal_slope']:.1f}°")
                    st.metric("Maximum Efficiency", f"{metrics['max_efficiency']:.3f}")
                with col2:
                    st.metric("Optimal Aspect", f"{metrics['optimal_aspect']:.1f}°")
                
                # Display the plot
                st.pyplot(fig)
                
                # Additional analysis
                st.subheader("Efficiency Analysis")
                efficiency_ratio = metrics['max_efficiency'] / model.solar_constant
                st.progress(float(efficiency_ratio))
                st.write(f"Relative Efficiency: {efficiency_ratio:.2%}")
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
    
    # Calculate current irradiance
    physics_model = SolarPhysicsIdeal()
    current_irradiance = physics_model.calculate_irradiance(
        latitude=torch.tensor([latitude]),
        time=torch.tensor([hour])
    )
    
    # Display current conditions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Conditions")
    st.sidebar.info(f"Estimated Irradiance: {current_irradiance.item():.2f} W/m²")

if __name__ == "__main__":
    main()