import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

MODEL_DIR = Path("../models")

def load_arima():
    return joblib.load(MODEL_DIR / "arima_model.pkl")

def load_prophet():
    return joblib.load(MODEL_DIR / "prophet_model.pkl")

def load_lstm():
    return load_model(MODEL_DIR / "lstm_model.h5")
