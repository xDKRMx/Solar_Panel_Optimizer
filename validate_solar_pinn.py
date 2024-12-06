import torch
import numpy as np
import matplotlib.pyplot as plt
from solar_pinn_ideal import SolarPINN
from physics_validator import SolarPhysicsIdeal, calculate_metrics

def load_best_model():
    """Load the best trained model."""
    model = SolarPINN()
    checkpoint = torch.load('best_solar_pinn_ideal.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_test_scenarios():
    """Generate various test scenarios to validate the model."""
    # Test case 1: Daily cycle at different latitudes
    latitudes = torch.tensor([-60, -30, 0, 30, 60])
    times = torch.linspace(0, 24, 48)
    day_of_year = torch.ones_like(times) * 172  # Summer solstice
    slope = torch.zeros_like(times)  # Flat surface
    aspect = torch.zeros_like(times)  # South-facing

    test_cases = []
    for lat in latitudes:
        lat_expanded = torch.full_like(times, lat)
        test_case = torch.stack([lat_expanded, times, day_of_year, slope, aspect], dim=1)
        test_cases.append(test_case)
    
    return torch.cat(test_cases, dim=0)

def validate_physical_constraints(model, physics_model, test_data):
    """Validate physical constraints of the model predictions."""
    with torch.no_grad():
        predictions = model(test_data)
        
        # 1. Check if predictions exceed solar constant
        max_pred = torch.max(predictions).item()
        print(f"\nPhysical Constraints Validation:")
        print(f"Maximum predicted irradiance: {max_pred:.2f} W/m² (should be ≤ 1367 W/m²)")
        
        # 2. Check day/night transitions
        lat, time = test_data[:, 0], test_data[:, 2]
        day_of_year = torch.floor(time / 24 * 365)
        hour_of_day = time % 24
        
        sunrise, sunset = model.boundary_conditions(lat, day_of_year)
        night_mask = (hour_of_day < sunrise) | (hour_of_day > sunset)
        night_predictions = predictions[night_mask]
        
        print(f"Mean nighttime irradiance: {torch.mean(night_predictions):.2f} W/m² (should be ≈ 0)")
        
        # 3. Calculate theoretical values for comparison
        theoretical_values = []
        for i in range(len(test_data)):
            irr = physics_model.calculate_irradiance(
                test_data[i, 0],  # latitude
                test_data[i, 1],  # time
                test_data[i, 3],  # slope
                test_data[i, 4]   # aspect
            )
            theoretical_values.append(irr)
        
        theoretical_values = torch.stack(theoretical_values).reshape(-1, 1)
        metrics = calculate_metrics(theoretical_values, predictions)
        
        print("\nValidation Metrics vs Theoretical Values:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MAE: {metrics['mae']:.2f} W/m²")
        print(f"RMSE: {metrics['rmse']:.2f} W/m²")
        
        return predictions, theoretical_values

def analyze_error_distribution(predictions, theoretical_values):
    """Analyze the error distribution of predictions."""
    errors = (predictions - theoretical_values).numpy()
    
    print("\nError Distribution Analysis:")
    print(f"Mean Error: {np.mean(errors):.2f} W/m²")
    print(f"Error Std Dev: {np.std(errors):.2f} W/m²")
    print(f"Error Range: [{np.min(errors):.2f}, {np.max(errors):.2f}] W/m²")

def main():
    # Load the trained model
    model = load_best_model()
    physics_model = SolarPhysicsIdeal()
    
    # Generate test scenarios
    print("Generating test scenarios...")
    test_data = generate_test_scenarios()
    
    # Validate physical constraints and get predictions
    print("\nValidating physical constraints...")
    predictions, theoretical = validate_physical_constraints(model, physics_model, test_data)
    
    # Analyze error distribution
    print("\nAnalyzing error distribution...")
    analyze_error_distribution(predictions, theoretical)

if __name__ == "__main__":
    main()
