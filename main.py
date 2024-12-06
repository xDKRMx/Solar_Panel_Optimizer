import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
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
    
    # Initialize location state
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 45.0
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 0.0

    # Create the map
    st.subheader("Select Location on Map")
    
    # Initialize the map centered on the current location
    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=3,
        tiles="OpenStreetMap"
    )
    
    # Add a marker for the current location
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup="Selected Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    
    # Display the map and get clicked coordinates
    map_data = st_folium(m, height=400, width=700)
    
    # Update coordinates if map is clicked
    if map_data['last_clicked']:
        st.session_state.latitude = map_data['last_clicked']['lat']
        st.session_state.longitude = map_data['last_clicked']['lng']
    
    # Location parameters with synchronized values
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.latitude = st.number_input(
            "Latitude",
            value=st.session_state.latitude,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            format="%.6f"
        )
    with col2:
        st.session_state.longitude = st.number_input(
            "Longitude",
            value=st.session_state.longitude,
            min_value=-180.0,
            max_value=180.0,
            step=0.1,
            format="%.6f"
        )
    
    # Display current location information
    st.sidebar.markdown(f"""
        **Selected Location:**  
        Latitude: {st.session_state.latitude:.6f}°  
        Longitude: {st.session_state.longitude:.6f}°
    """)
    
    # Time parameters
    day_of_year = st.sidebar.slider("Day of Year", 1, 365, 182)
    hour = st.sidebar.slider("Hour of Day", 0.0, 24.0, 12.0, 0.1)
    
    # Calculate predictions and metrics
    try:
        with torch.no_grad():
            current_input = torch.tensor([[
                st.session_state.latitude/90, st.session_state.longitude/180, 
                hour/24, 0/180,  # Default slope
                180/360  # Default aspect (south-facing)
            ]]).float()
            predicted_irradiance = model(current_input).item() * model.solar_constant
        
        # Calculate physics-based irradiance
        lat_tensor = torch.tensor([st.session_state.latitude], dtype=torch.float32)
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
        if accuracy_ratio > 85:
            metrics_color = "rgb(25, 52, 40)"   # Dark green for high accuracy
        elif accuracy_ratio > 50:
            metrics_color = "rgb(84, 52, 19)"   # Brown for medium accuracy
        else:
            metrics_color = "rgb(91, 47, 47)"   # Dark red for low accuracy
        
        # Display accuracy metrics with dynamic styling
        st.markdown("""
            <style>
            .metrics-box {
                padding: 10px;
                border-radius: 10px;
                margin: 10px 0;
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

    # Create 3D visualization
    if st.button("Generate 3D Surface Plot"):
        with st.spinner("Generating visualization..."):
            try:
                
                # Generate and display the 3D plot
                fig, metrics = create_surface_plot(model, latitude, longitude, hour)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display optimal parameters in green box
                st.markdown(f"""
                    <div class='accuracy-box'>
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
