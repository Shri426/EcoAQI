import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Paths
PROCESSED_PATH = "data/processed/"
MODEL_DIR = "data/processed/model/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load processed data

X = np.load(PROCESSED_PATH + "X_train.npy")
y = np.load(PROCESSED_PATH + "y_train.npy")

print("Loaded data shapes:", X.shape, y.shape)

if len(X) == 0:
    raise ValueError("X_train is empty. Cannot train model.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model (Green AI: lightweight)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully")
print("MAE:", round(mae, 2))
print("R² Score:", round(r2, 2))


# Save model
model_path = MODEL_DIR + "aqi_model.pkl"
joblib.dump(model, model_path)
print("Model saved at:", model_path)
