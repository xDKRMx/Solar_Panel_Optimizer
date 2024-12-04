import streamlit as st
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from solar_pinn import SolarPINN, train_pinn
from physics_equations import SolarRadiationPhysics

# Initialize PINN model and physics
model = SolarPINN()
physics = SolarRadiationPhysics()

# Initialize optimizer
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)

# Initialize model parameters
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 3)))  # 3 input features: latitude, longitude, time

def main():
    st.title("Solar Panel Placement Optimizer")
    st.write("Physics-Informed Neural Network for Optimal Solar Panel Placement")
    
    # Import required libraries
    import folium
    from streamlit_folium import st_folium
    
    # Create session states
    if 'selected_location' not in st.session_state:
        st.session_state.selected_location = {'lat': 0.0, 'lng': 0.0}
    if 'show_map' not in st.session_state:
        st.session_state.show_map = False
    
    def show_map_dialog():
        if st.session_state.show_map:
            with st.container():
                st.subheader("Location Selection Map")
                if st.button("Close Map"):
                    st.session_state.show_map = False
                    st.rerun()
                
                # Create base map centered at current location
                m = folium.Map(location=[st.session_state.selected_location['lat'], 
                                       st.session_state.selected_location['lng']], 
                              zoom_start=2)
                
                # Add existing marker if location is set
                if st.session_state.selected_location['lat'] != 0.0 or st.session_state.selected_location['lng'] != 0.0:
                    folium.Marker(
                        [st.session_state.selected_location['lat'], 
                         st.session_state.selected_location['lng']],
                        popup=f"Selected Location\nLat: {st.session_state.selected_location['lat']:.4f}\nLng: {st.session_state.selected_location['lng']:.4f}"
                    ).add_to(m)
                
                # Add click event handler
                map_data = st_folium(m, width=700, height=400)
                
                # Update selected location when map is clicked
                if map_data['last_clicked']:
                    st.session_state.selected_location = {
                        'lat': map_data['last_clicked']['lat'],
                        'lng': map_data['last_clicked']['lng']
                    }
                    # Recreate map with updated marker
                    m = folium.Map(location=[st.session_state.selected_location['lat'], 
                                           st.session_state.selected_location['lng']], 
                                  zoom_start=4)
                    folium.Marker(
                        [st.session_state.selected_location['lat'], 
                         st.session_state.selected_location['lng']],
                        popup=f"Selected Location\nLat: {st.session_state.selected_location['lat']:.4f}\nLng: {st.session_state.selected_location['lng']:.4f}"
                    ).add_to(m)
                    st_folium(m, width=700, height=400)
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    
    # Location parameters
    st.sidebar.subheader("Location Parameters")
    
    # Map selection button
    if st.sidebar.button("📍 Select Location on Map"):
        st.session_state.show_map = True
    
    # Show map dialog if button is clicked
    show_map_dialog()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, 
                                 value=st.session_state.selected_location['lat'], step=0.1,
                                 help="Click 'Select Location on Map' to choose location visually")
    with col2:
        latitude = st.slider("Latitude Slider", -90.0, 90.0, latitude)
    
    # Update session state if input fields change
    if latitude != st.session_state.selected_location['lat']:
        st.session_state.selected_location['lat'] = latitude

    col1, col2 = st.sidebar.columns(2)
    with col1:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, 
                                  value=st.session_state.selected_location['lng'], step=0.1,
                                  help="Click 'Select Location on Map' to choose location visually")
    with col2:
        longitude = st.slider("Longitude Slider", -180.0, 180.0, longitude)
    
    # Update session state if input fields change
    if longitude != st.session_state.selected_location['lng']:
        st.session_state.selected_location['lng'] = longitude
    
    # Time parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        day_number = st.number_input("Day of Year", min_value=1, max_value=365, value=182, step=1)
    with col2:
        day_number = st.slider("Day Slider", 1, 365, day_number)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        hour = st.number_input("Hour of Day", min_value=0, max_value=24, value=12, step=1)
    with col2:
        hour = st.slider("Hour Slider", 0, 24, hour)
    
    # Installation parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        slope = st.number_input("Panel Slope", min_value=0.0, max_value=90.0, value=30.0, step=0.1)
    with col2:
        slope = st.slider("Slope Slider", 0.0, 90.0, slope)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        aspect = st.number_input("Panel Aspect", min_value=0.0, max_value=360.0, value=180.0, step=0.1)
    with col2:
        aspect = st.slider("Aspect Slider", 0.0, 360.0, aspect)
    
    # Atmospheric parameters
    st.sidebar.subheader("Atmospheric Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        atm_transmission = st.number_input("Atmospheric Transmission", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col2:
        atm_transmission = st.slider("Transmission Slider", 0.0, 1.0, atm_transmission)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        cloud_cover = st.number_input("Cloud Cover", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with col2:
        cloud_cover = st.slider("Cloud Cover Slider", 0.0, 1.0, cloud_cover)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        wavelength = st.number_input("Wavelength (μm)", min_value=0.3, max_value=1.0, value=0.5, step=0.01)
    with col2:
        wavelength = st.slider("Wavelength Slider", 0.3, 1.0, wavelength)
    
    # Create input data
    input_data = jnp.array([[latitude, longitude, hour]])
    
    # Model prediction using PINN
    prediction = model.apply(params, input_data)
    
    # Calculate physics-based solar position
    zenith_angle = physics.solar_position_equations(latitude, longitude, hour)
    
    # Calculate irradiance using physics equations
    physics_irradiance = physics.solar_constant * jnp.cos(zenith_angle) * \
                        (1 - physics.compute_absorption_coefficient(0))
    
    # Calculate accuracy metrics
    relative_error = abs(prediction.item() - physics_irradiance) / physics_irradiance if physics_irradiance > 0 else float('inf')
    accuracy_ratio = 1 - relative_error

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PINN Prediction")
        st.write(f"Predicted Irradiance: {prediction.item():.2f} W/m²")
    
    with col2:
        st.subheader("Physics Calculation")
        st.write(f"Physics-based Irradiance: {physics_irradiance:.2f} W/m²")
    
    # Display accuracy metrics with color coding
    st.subheader("Accuracy Metrics")
    if accuracy_ratio > 0.9:
        st.success(f"Accuracy Ratio: {accuracy_ratio:.2%}")
    elif accuracy_ratio > 0.7:
        st.warning(f"Accuracy Ratio: {accuracy_ratio:.2%}")
    else:
        st.error(f"Accuracy Ratio: {accuracy_ratio:.2%}")
    
    st.write(f"Relative Error: {relative_error:.2%}")
    
    # Optimization analysis
    if st.button("Run Optimization Analysis"):
        # Create time points for analysis
        hours = jnp.linspace(0, 24, 24)
        
        # Calculate daily irradiance profile
        irradiance_profile = jnp.zeros(24)
        for i, hour in enumerate(hours):
            input_data = jnp.array([[latitude, longitude, hour]])
            prediction = model.apply(params, input_data)
            irradiance_profile = irradiance_profile.at[i].set(prediction[0, 0])
        
        # Calculate optimal tilt and orientation
        zenith_angles = jnp.array([physics.solar_position_equations(latitude, longitude, h) for h in hours])
        optimal_tilt = jnp.rad2deg(jnp.mean(zenith_angles))
        optimal_orientation = 180.0  # South-facing in Northern hemisphere, North-facing in Southern
        
        # Display results
        st.subheader("Daily Solar Irradiance Profile")
        st.line_chart(irradiance_profile)
        
        st.success(f"""
        Optimal Parameters:
        - Tilt Angle: {optimal_tilt:.1f}°
        - Orientation: {optimal_orientation:.1f}° ({optimal_orientation <= 180 and 'South' or 'North'}-facing)
        - Peak Irradiance: {jnp.max(irradiance_profile):.2f} W/m²
        - Daily Average: {jnp.mean(irradiance_profile):.2f} W/m²
        """)

if __name__ == "__main__":
    main()
