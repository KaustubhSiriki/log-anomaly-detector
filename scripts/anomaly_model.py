import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "system_logs.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    # Assumes the CSV file has no header and columns in the order:
    # event, cpu_usage, memory_usage, disk_io, network_traffic, response_time
    logs = pd.read_csv(LOG_FILE, names=["event", "cpu_usage", "memory_usage", "disk_io", "network_traffic", "response_time"])
    return logs

def preprocess_data(logs):
    # Map event names to numerical labels
    attack_types = {
        'normal': 0,
        'malware': 1,
        'ddos': 2,
        'brute_force': 3,
        'ransomware': 4,
        'port_scan': 5
    }
    logs["event_encoded"] = logs["event"].map(attack_types)
    feature_cols = ["cpu_usage", "memory_usage", "disk_io", "network_traffic", "response_time"]

    # Scale the features
    scaler = MinMaxScaler()
    logs[feature_cols] = scaler.fit_transform(logs[feature_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    X = logs[feature_cols].values
    y = logs["event_encoded"].values
    # Convert labels to one-hot encoding for better training
    y_cat = to_categorical(y, num_classes=len(attack_types))
    return X, y_cat

def train_isolation_forest(X):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, os.path.join(MODEL_DIR, "isolation_forest.pkl"))

def train_lstm(X, y_cat):
    # Reshape X for LSTM: (samples, timesteps, features) where timesteps=1
    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, X.shape[1])),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Split into training and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)
    
    # Early stopping callback to prevent overfitting
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stop])
    
    model.save(os.path.join(MODEL_DIR, "lstm_classifier.h5"))

if __name__ == "__main__":
    logs = load_data()
    X, y_cat = preprocess_data(logs)
    train_isolation_forest(X)
    train_lstm(X, y_cat)
