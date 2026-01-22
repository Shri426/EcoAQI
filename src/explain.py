import joblib

# Load trained model
model = joblib.load("data/processed/model/aqi_model.pkl")

features = ["temperature", "humidity", "wind_speed"]
coefficients = model.coef_

print("🔍 AQI Influence Explanation:")
for f, c in zip(features, coefficients):
    effect = "increases AQI" if c > 0 else "reduces AQI"
    print(f"- {f}: {effect} (weight = {round(c, 2)})")

