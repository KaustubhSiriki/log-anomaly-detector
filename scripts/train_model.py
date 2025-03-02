import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os

# Ensure model directories exist
os.makedirs("models", exist_ok=True)

# Load sample log data (replace with real logs later)
def generate_sample_logs(n_samples=1000):
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq='H')
    cpu_usage = np.random.normal(50, 10, n_samples)  # Normal CPU usage
    mem_usage = np.random.normal(40, 8, n_samples)
    disk_io = np.random.normal(100, 25, n_samples)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    
    # Inject anomalies
    cpu_usage[anomaly_indices] = np.random.uniform(80, 100, len(anomaly_indices))
    mem_usage[anomaly_indices] = np.random.uniform(70, 90, len(anomaly_indices))
    disk_io[anomaly_indices] = np.random.uniform(200, 300, len(anomaly_indices))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'mem_usage': mem_usage,
        'disk_io': disk_io
    })
    return df

data = generate_sample_logs()

# Feature Engineering
scaler = MinMaxScaler()
X = scaler.fit_transform(data[['cpu_usage', 'mem_usage', 'disk_io']])
joblib.dump(scaler, "models/scaler.pkl")

# === Train Isolation Forest Model ===
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X)
joblib.dump(iso_forest, "models/isolation_forest.pkl")

# === Train LSTM Autoencoder Model ===
time_steps = 10  # Lookback window
X_series = np.array([X[i-time_steps:i] for i in range(time_steps, len(X))])

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, 3)),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(3)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_series, X[time_steps:], epochs=20, batch_size=32, verbose=1)
model.save("models/lstm_autoencoder.h5")

print("Models trained and saved!")
