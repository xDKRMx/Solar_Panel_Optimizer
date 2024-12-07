import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from solar_pinn_ideal import SolarPINN
from physics_validator import SolarPhysicsIdeal
from visualize_results import create_surface_plot, load_model


def format_time(decimal_hour):
    """Convert decimal hour to HH:MM format."""
    hours = int(decimal_hour)
    minutes = int((decimal_hour % 1) * 60)
    return f"{hours:02d}:{minutes:02d}"


def time_to_decimal(hours, minutes):
    """Convert hours and minutes to decimal hours."""
    return hours + (minutes / 60)


def main():
    st.title("Solar Panel Placement Optimizer")
    st.write(
        "Physics-Informed Neural Network for Optimal Solar Panel Placement")

    def update_param(param_name):
        if f'{param_name}_slider' in st.session_state and f'{param_name}_input' in st.session_state:
            value = st.session_state[f'{param_name}_slider']
            st.session_state[f'{param_name}_input'] = value
            st.session_state[param_name] = value

    # Initialize session state
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 45.0
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 0.0
    if 'show_map' not in st.session_state:
        st.session_state.show_map = False
    if 'day_of_year' not in st.session_state:
        st.session_state.day_of_year = 182
    if 'hour' not in st.session_state:
        st.session_state.hour = 12.0

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

    if st.sidebar.button("📍 Select Location on Map", key="location_button"):
        st.session_state.show_map = not st.session_state.show_map

    # Location parameters section
    st.sidebar.subheader("Location Parameters")

    # Latitude input with slider and number input
    lat_col1, lat_col2 = st.sidebar.columns([3, 1])
    with lat_col1:
        latitude = st.slider("Latitude",
                             -90.0,
                             90.0,
                             value=st.session_state.get('latitude', 45.0),
                             key='latitude_slider',
                             on_change=lambda: update_param('latitude'))
    with lat_col2:
        st.write("")
        latitude = st.number_input("Latitude Value",
                                   -90.0,
                                   90.0,
                                   value=st.session_state.get(
                                       'latitude', 45.0),
                                   key='latitude_input',
                                   label_visibility="collapsed",
                                   on_change=lambda: update_param('latitude'))

    # Longitude input with slider and number input
    lon_col1, lon_col2 = st.sidebar.columns([3, 1])
    with lon_col1:
        longitude = st.slider("Longitude",
                              -180.0,
                              180.0,
                              value=st.session_state.get('longitude', 0.0),
                              key='longitude_slider',
                              on_change=lambda: update_param('longitude'))
    with lon_col2:
        st.write("")
        longitude = st.number_input(
            "Longitude Value",
            -180.0,
            180.0,
            value=st.session_state.get('longitude', 0.0),
            key='longitude_input',
            label_visibility="collapsed",
            on_change=lambda: update_param('longitude'))

    # Day of Year input with slider and number input
    day_col1, day_col2 = st.sidebar.columns([3, 1])
    with day_col1:
        day_of_year = st.slider("Day of Year",
                                1,
                                365,
                                value=st.session_state.get('day_of_year', 182),
                                key='day_of_year_slider',
                                on_change=lambda: update_param('day_of_year'))
    with day_col2:
        st.write("")
        day_of_year = st.number_input(
            "Day of Year Value",
            1,
            365,
            value=st.session_state.get('day_of_year', 182),
            key='day_of_year_input',
            label_visibility="collapsed",
            on_change=lambda: update_param('day_of_year'))

    # Hour of Day input with slider and number input
    def generate_time_range():
        """Generate a list of time values in HH:MM format from 00:00 to 23:59."""
        times = []
        for hour in range(24):
            for minute in range(0, 60, 1):  # Step by 1 minute
                times.append(f"{hour:02}:{minute:02}")
        return times

    def parse_time(selected_time):
        """Parse the selected time string into hours and minutes."""
        hour, minute = map(int, selected_time.split(":"))
        return hour + minute / 60

    # Generate time options for the slider
    time_range = generate_time_range()

    # Sidebar layout
    hour_col1, hour_col2 = st.sidebar.columns([3, 1])

    with hour_col1:
        # Slider with formatted time options
        selected_time = st.select_slider(
            "Hour of Day",
            options=time_range,  # Use the generated time range
            value="12:00",  # Default to noon
        )
        st.session_state['hour'] = parse_time(
            selected_time)  # Convert to decimal hour
 
    with hour_col2:
        # Allow user to type directly
        selected_time_input = st.text_input(
            "",
            value=selected_time,  # Default matches slider
        )
        # Validate and update
        if selected_time_input in time_range:
            selected_time = selected_time_input
            st.session_state['hour'] = parse_time(selected_time)
        else:
            st.error("Invalid time format. Please use HH:MM.")

    # Update session state when values change
    if latitude != st.session_state.get('latitude'):
        st.session_state['latitude'] = latitude
    if longitude != st.session_state.get('longitude'):
        st.session_state['longitude'] = longitude

    # Add map section before predictions
    if st.session_state.show_map:
        st.subheader("Select Location on Map")
        m = folium.Map(location=[latitude, longitude],
                       zoom_start=3,
                       tiles="OpenStreetMap")

        folium.Marker(
            [
                st.session_state.get('latitude', latitude),
                st.session_state.get('longitude', longitude)
            ],
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        map_data = st_folium(m, height=400, width=700, key="map")

        if map_data['last_clicked'] is not None:
            st.session_state['latitude'] = map_data['last_clicked']['lat']
            st.session_state['longitude'] = map_data['last_clicked']['lng']
            st.success(
                f"Location selected: {st.session_state['latitude']:.4f}°, {st.session_state['longitude']:.4f}°"
            )
            st.rerun()

    # Calculate predictions and metrics
    try:
        with torch.no_grad():
            hour = st.session_state['hour']
            # Normalize day_of_year to [0,1] range
            normalized_day = (day_of_year - 1) / 364  # Maps [1,365] to [0,1]
            
            current_input = torch.tensor([[
                latitude / 90,
                longitude / 180,
                hour / 24,
                normalized_day,
                0 / 180,  # Default slope
                180 / 360  # Default aspect (south-facing)
            ]]).float()
            predicted_irradiance = model(
                current_input).item() * model.solar_constant

        # Calculate physics-based irradiance and efficiency
        lat_tensor = torch.tensor([latitude], dtype=torch.float32)
        hour_tensor = torch.tensor([hour], dtype=torch.float32)
        physics_irradiance = physics_model.calculate_irradiance(
            latitude=lat_tensor, time=hour_tensor).item()

        # Calculate panel efficiency
        efficiency = physics_model.calculate_efficiency(
            latitude=lat_tensor,
            time=hour_tensor,
            slope=torch.tensor([0.0]),  # Default horizontal
            panel_azimuth=torch.tensor([180.0])  # Default south-facing
        ).item()

        # Display predictions section
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PINN Prediction")
            st.write(f"Predicted Irradiance: {predicted_irradiance:.2f} W/m²")
        with col2:
            st.subheader("Physics Calculation")
            st.write(
                f"Physics-based Irradiance: {physics_irradiance:.2f} W/m²")

        # Calculate accuracy metrics with new formula and error handling
        try:
            max_irradiance = max(predicted_irradiance, physics_irradiance)
            if max_irradiance == 0:
                accuracy_ratio = 100.0  # Set to 100% when both values are 0
            else:
                accuracy_ratio = (max_irradiance - abs(physics_irradiance - predicted_irradiance)) / max_irradiance * 100
        except ZeroDivisionError:
            accuracy_ratio = 100.0  # Handle any other division by zero cases

        # Calculate relative error only when physics_irradiance is not zero
        try:
            relative_error = abs(predicted_irradiance - physics_irradiance) / physics_irradiance * 100 if physics_irradiance != 0 else 0.0
        except ZeroDivisionError:
            relative_error = 0.0  # Handle division by zero for relative error

        # Determine color based on accuracy ratio
        if accuracy_ratio > 85:
            metrics_color = "rgb(25, 52, 40)"  # Dark green for high accuracy
        elif accuracy_ratio > 50:
            metrics_color = "rgb(84, 52, 19)"  # Brown for medium accuracy
        else:
            metrics_color = "rgb(91, 47, 47)"  # Dark red for low accuracy

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
        """,
                    unsafe_allow_html=True)

        st.markdown(f"""
            <div class='metrics-box' style='background-color: {metrics_color};'>
                <div class='metrics-content'>
                    <p class='metrics-header'>Accuracy Metrics</p>
                    <p>Accuracy Ratio: {accuracy_ratio:.2f}%</p>
                    <p>Relative Error: {relative_error:.2f}%</p>
                </div>
            </div>
        """,
                    unsafe_allow_html=True)

        # Create 3D visualization
        plot_container = st.container()
        if st.button("Generate 3D Surface Plot"):
            with plot_container:
                with st.spinner("Generating visualization..."):
                    try:
                        fig, metrics = create_surface_plot(
                            model, st.session_state.get('latitude', latitude),
                            st.session_state.get('longitude', longitude), hour)
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(f'''
                            <div class='accuracy-box'>
                                <h3>Optimal Parameters:</h3>
                                <ul>
                                    <li>Slope: {metrics['optimal_slope']:.1f}°</li>
                                    <li>Aspect: {metrics['optimal_aspect']:.1f}°</li>
                                    <li>Expected Efficiency: {metrics['max_efficiency']:.3f}</li>
                                </ul>
                            </div>
                        ''',
                                    unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")

    except Exception as e:
        st.error(f"Error calculating predictions: {str(e)}")


if __name__ == "__main__":
    main()
