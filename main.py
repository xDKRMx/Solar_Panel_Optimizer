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
    
    # Initialize physics model at the start
    physics_model = SolarPhysicsIdeal()
    
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
    hour = st.sidebar.slider("Hour of Day", 0.0, 24.0, 12.0, 0.1)
    
    # Create 3D visualization
    if st.button("Generate 3D Surface Plot"):
        with st.spinner("Generating visualization..."):
            try:
                fig, metrics = create_surface_plot(model, latitude, longitude, hour)
                
                # Display predicted and physics-based values
                st.subheader("Solar Panel Performance Metrics")
                
                # Calculate current predicted and physics-based values
                with torch.no_grad():
                    current_input = torch.tensor([[
                        latitude/90, longitude/180, 
                        hour/24, metrics['optimal_slope']/180, 
                        metrics['optimal_aspect']/360
                    ]]).float()
                    predicted_irradiance = model(current_input).item() * model.solar_constant
                
                try:
                    # Convert inputs to tensors with proper dtype
                    lat_tensor = torch.tensor([latitude], dtype=torch.float32)
                    hour_tensor = torch.tensor([hour], dtype=torch.float32)
                    
                    current_irradiance = physics_model.calculate_irradiance(
                        latitude=lat_tensor,
                        time=hour_tensor
                    ).item()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PINN Predicted Irradiance", f"{predicted_irradiance:.1f} W/m²")
                        st.metric("Optimal Slope", f"{metrics['optimal_slope']:.1f}°")
                    with col2:
                        st.metric("Physics-based Irradiance", f"{current_irradiance:.1f} W/m²")
                        st.metric("Optimal Aspect", f"{metrics['optimal_aspect']:.1f}°")
                except Exception as e:
                    st.error(f"Error calculating irradiance: {str(e)}")
                
                # Display the plot
                st.pyplot(fig)
                
                # Efficiency metrics section
                st.subheader("Efficiency Analysis")
                efficiency_ratio = metrics['max_efficiency']
                st.progress(float(efficiency_ratio))
                st.write(f"Maximum Efficiency: {efficiency_ratio:.1%}")
                
                # Display relative efficiency at current settings
                relative_efficiency = predicted_irradiance / (model.solar_constant * efficiency_ratio)
                st.write(f"Relative Efficiency at Current Settings: {relative_efficiency:.1%}")
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
    
    # Calculate current irradiance using physics model
    try:
        current_irradiance = physics_model.calculate_irradiance(
            latitude=torch.tensor([latitude]),
            time=torch.tensor([hour])
        )
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Conditions")
        st.sidebar.info(f"Estimated Irradiance: {current_irradiance.item():.2f} W/m²")
    except Exception as e:
        st.sidebar.error(f"Error calculating current irradiance: {str(e)}")

if __name__ == "__main__":
    main()