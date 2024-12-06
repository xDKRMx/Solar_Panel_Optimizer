import streamlit as st
import torch
from solar_pinn_ideal import SolarPINN
from physics_validator import SolarPhysicsIdeal
from visualize_results import create_surface_plot, load_model
from visualization.plots import SolarPlotter

def main():
    st.title("Solar Panel Placement Optimizer")
    st.write("Physics-Informed Neural Network for Optimal Solar Panel Placement")
    
    # Initialize physics model and plotter
    physics_model = SolarPhysicsIdeal()
    plotter = SolarPlotter()
    
    # Load the trained model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
        
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Local Analysis", "Global Map"])
    
    # Sidebar inputs
    st.sidebar.header("Parameters")
    
    # Time parameters (shared between tabs)
    day_of_year = st.sidebar.slider("Day of Year", 1, 365, 182)
    hour = st.sidebar.slider("Hour of Day", 0.0, 24.0, 12.0, 0.1)
    
    # Local Analysis Tab
    with tab1:
        # Location parameters for local analysis
        latitude = st.slider("Latitude", -90.0, 90.0, 45.0, 0.1)
        longitude = st.slider("Longitude", -180.0, 180.0, 0.0, 0.1)
        
        # Calculate predictions and metrics
        try:
            with torch.no_grad():
                current_input = torch.tensor([[
                    latitude/90, longitude/180, 
                    hour/24, 0/180, 180/360  # Default slope and aspect
                ]], dtype=torch.float32)
                
                # Generate visualization
                fig, metrics = create_surface_plot(
                    model, latitude, longitude, hour
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                st.markdown("""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h3>Optimal Parameters:</h3>
                        <ul>
                            <li>Slope: {:.1f}°</li>
                            <li>Aspect: {:.1f}°</li>
                            <li>Expected Efficiency: {:.3f}</li>
                        </ul>
                    </div>
                """.format(
                    metrics['optimal_slope'],
                    metrics['optimal_aspect'],
                    metrics['max_efficiency']
                ), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
    
    # Global Map Tab
    with tab2:
        st.subheader("Global Solar Irradiance Map")
        st.write("This map shows the predicted solar irradiance across the globe for the selected time.")
        
        if st.button("Generate Global Map"):
            with st.spinner("Generating global irradiance map..."):
                try:
                    # Create world map visualization
                    world_map = plotter.create_world_irradiance_map(
                        model=model,
                        hour=hour,
                        day_of_year=day_of_year
                    )
                    
                    # Display the map
                    st.plotly_chart(world_map, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating world map: {str(e)}")

if __name__ == "__main__":
    main()