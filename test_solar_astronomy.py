from utils.solar_astronomy import SolarAstronomyCalculator
import datetime

def test_solar_calculations():
    calculator = SolarAstronomyCalculator()
    
    # Test for San Francisco (37.7749° N, 122.4194° W) on winter solstice (day 355)
    latitude = 37.7749
    longitude = -122.4194
    day = 355  # December 21st (approximately)
    timezone_offset = -8  # PST
    
    sunrise, sunset = calculator.calculate_sunrise_sunset(latitude, day, timezone_offset, longitude)
    
    print(f"Test Results for San Francisco (37.7749° N) on Winter Solstice:")
    print(f"Sunrise: {calculator.format_time(sunrise)}")
    print(f"Sunset: {calculator.format_time(sunset)}")
    print(f"Day length: {calculator.format_time(sunset - sunrise)}")
    
    # Test for current date
    current_day = calculator.get_day_of_year()
    sunrise_today, sunset_today = calculator.calculate_sunrise_sunset(latitude, current_day, timezone_offset, longitude)
    
    print(f"\nTest Results for San Francisco Today:")
    print(f"Day of year: {current_day}")
    print(f"Sunrise: {calculator.format_time(sunrise_today)}")
    print(f"Sunset: {calculator.format_time(sunset_today)}")
    print(f"Day length: {calculator.format_time(sunset_today - sunrise_today)}")

if __name__ == "__main__":
    test_solar_calculations()
