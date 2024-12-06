import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from solar_pinn_ideal import SolarPINN
from physics_validator import SolarPhysicsIdeal
from visualize_results import create_surface_plot, load_model

def main():
    # Configure the page
    st.set_page_config(
        page_title="Solar Panel Placement Optimizer",
        page_icon="☀️",
        layout="wide"
    )
    
    st.title("Solar Panel Placement Optimizer")
    st.write("Physics-Informed Neural Network for Optimal Solar Panel Placement")

    # Initialize models
    try:
        physics_model = SolarPhysicsIdeal()
        model = load_model('best_solar_pinn_ideal.pth')
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        st.stop()
        return

    # Sidebar inputs
    st.sidebar.header("Parameters")
    latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 45.0, 0.1)
    longitude = st.sidebar.slider("Longitude", -180.0, 180.0, 0.0, 0.1)
    day_of_year = st.sidebar.slider("Day of Year", 1, 365, 182)
    hour = st.sidebar.slider("Hour of Day", 0.0, 24.0, 12.0, 0.1)

    # Calculate predictions and metrics
    try:
        with torch.no_grad():
            # Calculate seasonal weight
            day_of_year = torch.floor(hour / 24 * 365)
            seasonal_weight = torch.abs(torch.sin(2 * torch.pi * day_of_year / 365))
            
            current_input = torch.tensor([[
                latitude/90, longitude/180, 
                hour/24, 0/180,  # Default slope
                180/360,  # Default aspect (south-facing)
                seasonal_weight  # Add seasonal weight
            ]], dtype=torch.float32)
            predicted_irradiance = model(current_input).item() * model.solar_constant

        # Calculate physics-based irradiance
        lat_tensor = torch.tensor([latitude], dtype=torch.float32)
        hour_tensor = torch.tensor([hour], dtype=torch.float32)
        physics_irradiance = physics_model.calculate_irradiance(
            latitude=lat_tensor,
            time=hour_tensor
        ).item()

        # Display predictions section
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PINN Prediction")
            st.write(f"Predicted Irradiance: {predicted_irradiance:.2f} W/m²")
        with col2:
            st.subheader("Physics Calculation")
            st.write(f"Physics-based Irradiance: {physics_irradiance:.2f} W/m²")

        # Calculate accuracy metrics
        accuracy_ratio = min(predicted_irradiance, physics_irradiance) / max(predicted_irradiance, physics_irradiance) * 100
        relative_error = abs(predicted_irradiance - physics_irradiance) / physics_irradiance * 100

        # Determine color based on accuracy ratio
        if accuracy_ratio == 0:
            metrics_color = "rgb(91, 47, 47)"  # Dark red for 0% accuracy
        elif accuracy_ratio < 50:
            metrics_color = "rgb(91, 47, 47)"  # Dark red for very low accuracy
        elif accuracy_ratio < 80:
            metrics_color = "rgb(84, 52, 19)"  # Brown for moderate accuracy
        else:
            metrics_color = "rgb(27, 67, 50)"  # Dark green for high accuracy

        # Display accuracy metrics with dynamic styling
        st.markdown("""
            <style>
            .metrics-box {
                padding: 8px;
                border-radius: 8px;
                margin: 8px 0;
            }
            .metrics-content {
                margin: 0;
                font-size: 0.9em;
            }
            .metrics-header {
                margin: 0 0 8px 0;
                font-size: 1.1em;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='metrics-box' style='background-color: {metrics_color};'>
                <div class='metrics-content'>
                    <p class='metrics-header'>Accuracy Metrics</p>
                    <p>Accuracy Ratio: {accuracy_ratio:.2f}%</p>
                    <p>Relative Error: {relative_error:.2f}%</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error calculating predictions: {str(e)}")
        return

    # Create 3D visualization
    if st.button("Generate 3D Surface Plot"):
        with st.spinner("Generating visualization..."):
            try:
                fig, metrics = create_surface_plot(model, latitude, longitude, hour)
                st.plotly_chart(fig, use_container_width=True)

                # Display optimal parameters
                st.markdown("""
                    <style>
                    .optimal-box {
                        background-color: rgb(27, 67, 50);
                        padding: 8px;
                        border-radius: 8px;
                        margin: 8px 0;
                        color: white;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class='optimal-box'>
                        <h3>Optimal Parameters:</h3>
                        <ul>
                            <li>Slope: {metrics['optimal_slope']:.1f}°</li>
                            <li>Aspect: {metrics['optimal_aspect']:.1f}°</li>
                            <li>Expected Efficiency: {metrics['max_efficiency']:.3f}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    main()
