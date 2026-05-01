# -*- coding: utf-8 -*-
import os
import logging
import joblib
import pathlib as path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = path.Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# Load the trained model only once (module-level caching)
model = joblib.load(MODEL_PATH)
logging.info("Model loaded successfully for prediction.")

def predict_heart_disease(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = int(model.predict(df)[0])
    logging.info(f"Prediction for heart disease: {prediction}")
    probability = float(model.predict_proba(df)[0][1])
    logging.info(f"Predicted probability of heart disease: {probability:.4f}")
    return prediction, probability
