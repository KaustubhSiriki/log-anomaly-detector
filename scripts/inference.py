from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load Models
scaler = joblib.load("models/scaler.pkl")
iso_forest = joblib.load("models/isolation_forest.pkl")
lstm_model = load_model("models/lstm_autoencoder.h5", custom_objects={"mse": MeanSquaredError()})

# Define Input Model
class LogData(BaseModel):
    cpu_usage: float
    mem_usage: float
    disk_io: float

@app.get("/")
def home():
    return {"message": "Log Anomaly Detection API is running!"}

@app.post("/predict/")
def predict(log: LogData):
    data = np.array([[log.cpu_usage, log.mem_usage, log.disk_io]])
    scaled_data = scaler.transform(data)

    # Isolation Forest Prediction
    iso_pred = iso_forest.predict(scaled_data)[0]  # -1 = anomaly, 1 = normal

    # LSTM Autoencoder Reconstruction Error
    lstm_pred = lstm_model.predict(scaled_data.reshape(1, 1, 3))
    lstm_error = np.mean(np.abs(lstm_pred - scaled_data))

    # Threshold for LSTM-based anomaly detection
    lstm_threshold = 0.1  # Adjust based on training data

    anomaly_detected = iso_pred == -1 or lstm_error > lstm_threshold

    return {
        "cpu_usage": log.cpu_usage,
        "mem_usage": log.mem_usage,
        "disk_io": log.disk_io,
        "isolation_forest_anomaly": bool(iso_pred == -1),
        "lstm_anomaly": bool(lstm_error > lstm_threshold),
        "overall_anomaly": bool(anomaly_detected),
    }

# Run API only if script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
