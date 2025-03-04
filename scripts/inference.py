import os
import sys
import logging
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from .database import SessionLocal, LogEntry
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Determine project root and models directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Load API Key from environment
API_KEY = os.getenv("API_KEY")

# Configure logging to a file
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("API started successfully")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Can restrict to frontend URL later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained models and scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
isolation_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_classifier.h5"))

# Define threat scores and event label mapping
THREAT_SCORES = {
    "normal": 0,
    "malware": 90,
    "ddos": 85,
    "brute_force": 70,
    "ransomware": 95,
    "port_scan": 65
}
EVENT_LABELS = {
    0: "normal",
    1: "malware",
    2: "ddos",
    3: "brute_force",
    4: "ransomware",
    5: "port_scan"
}

class LogInput(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_traffic: float
    response_time: float

# Preprocess incoming log data: scale and reshape for LSTM
def preprocess_log(log: LogInput):
    try:
        data = np.array([[log.cpu_usage, log.memory_usage, log.disk_io, log.network_traffic, log.response_time]])
        data = scaler.transform(data)
        data = data.reshape((data.shape[0], 1, data.shape[1]))
        return data
    except Exception as e:
        logging.error("Preprocessing error: " + str(e))
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

@app.post("/log/")
@limiter.limit("10/minute")
def analyze_log(request: Request, log: LogInput, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    try:
        data = preprocess_log(log)
        # For Isolation Forest, use 2D data
        data_if = data.reshape((data.shape[0], data.shape[2]))
        iso_pred = isolation_forest.predict(data_if)[0] == -1  # True if considered an outlier
        
        # LSTM classifier prediction
        lstm_pred = lstm_model.predict(data)
        predicted_label_index = int(np.argmax(lstm_pred, axis=1)[0])
        event_type = EVENT_LABELS[predicted_label_index]
        risk_score = THREAT_SCORES[event_type]
    except Exception as e:
        logging.error("Prediction error: " + str(e))
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    try:
        session = SessionLocal()
        new_entry = LogEntry(
            timestamp=datetime.utcnow(),
            event=event_type,
            cpu_usage=log.cpu_usage,
            memory_usage=log.memory_usage,
            disk_io=log.disk_io,
            network_traffic=log.network_traffic,
            response_time=log.response_time,
            risk_score=risk_score,
        )
        session.add(new_entry)
        session.commit()
        session.close()
    except Exception as e:
        logging.error("Database error: " + str(e))
        raise HTTPException(status_code=500, detail="Database error")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "cpu_usage": log.cpu_usage,
        "memory_usage": log.memory_usage,
        "disk_io": log.disk_io,
        "network_traffic": log.network_traffic,
        "response_time": log.response_time,
        "isolation_forest_anomaly": bool(iso_pred),
        "risk_score": int(risk_score)
    }

@app.get("/")
def root():
    return {"message": "Log Anomaly Detection API is running!"}
