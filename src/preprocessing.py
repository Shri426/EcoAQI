import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load raw data
aqi_df = pd.read_csv(RAW_PATH + "ground_stations.csv")
weather_df = pd.read_csv(RAW_PATH + "weather_data.csv")


# Standardize column names
aqi_df.columns = aqi_df.columns.str.lower().str.strip()
weather_df.columns = weather_df.columns.str.lower().str.strip()

print("AQI columns:", aqi_df.columns.tolist())
print("Weather columns:", weather_df.columns.tolist())


# Identify AQI column SAFELY
aqi_col = None
for col in aqi_df.columns:
    if "aqi" in col:
        aqi_col = col
        break

if aqi_col is None:
    raise ValueError(" No AQI column found in ground_stations.csv")


# Convert AQI to numeric (SAFE)
aqi_series = aqi_df[aqi_col]

# Ensure it is a Series (not DataFrame)
if isinstance(aqi_series, pd.DataFrame):
    aqi_series = aqi_series.iloc[:, 0]

aqi_numeric = pd.to_numeric(aqi_series, errors="coerce")
aqi_numeric = aqi_numeric.dropna()


# Handle weather data

weather_map = {
    "temp": "temperature",
    "temperature": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind_speed",
    "wind-speed": "wind_speed",
    "windspeed": "wind_speed"
}

for col in list(weather_df.columns):
    if col in weather_map:
        weather_df.rename(columns={col: weather_map[col]}, inplace=True)

required_weather_cols = ["temperature", "humidity", "wind_speed"]
weather_df = weather_df[required_weather_cols].dropna()

# Align dataset lengths
min_len = min(len(aqi_numeric), len(weather_df))
aqi_numeric = aqi_numeric.iloc[:min_len]
weather_df = weather_df.iloc[:min_len]


# Final NumPy arrays
X_train = weather_df.to_numpy(dtype=float)
y_train = aqi_numeric.to_numpy(dtype=float)


# Save processed data
np.save(PROCESSED_PATH + "X_train.npy", X_train)
np.save(PROCESSED_PATH + "y_train.npy", y_train)

print("Preprocessing completed successfully")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("AQI dtype:", y_train.dtype)
