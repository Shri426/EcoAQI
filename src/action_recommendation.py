def aqi_action(aqi):
    if aqi <= 50:
        return "Good air quality. No action needed."
    elif aqi <= 100:
        return "Moderate air quality. Sensitive individuals should limit outdoor activity."
    elif aqi <= 200:
        return "Unhealthy air. Reduce outdoor exposure. Wear masks if needed."
    elif aqi <= 300:
        return "Very unhealthy air. Avoid outdoor activity. Issue public health alerts."
    else:
        return "Hazardous air. Emergency response required."

# Example usage
sample_aqi = 180
print("AQI:", sample_aqi)
print("Action:", aqi_action(sample_aqi))
