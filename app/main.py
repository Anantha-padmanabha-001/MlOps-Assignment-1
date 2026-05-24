# =============================================================
# FastAPI Prediction Server — Heart Disease MLOps API
# =============================================================
# This file creates a web API with a /predict endpoint.
# It loads the trained model and returns predictions as JSON.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import logging

# Set up logging — records every request to a log file
logging.basicConfig(
    filename="api_requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts risk of heart disease using ML model",
    version="1.0.0"
)

# Load model and feature columns on startup
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# =============================================================
# INPUT SCHEMA
# =============================================================
# Defines exactly what JSON input the API expects
# Each field matches a column in the Heart Disease dataset

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# =============================================================
# ENDPOINTS
# =============================================================

@app.get("/")
def home():
    """Health check endpoint"""
    return {"status": "running", "message": "Heart Disease API is live!"}

@app.post("/predict")
def predict(patient: PatientData):
    """
    Accepts patient data as JSON and returns:
    - prediction: 0 (no disease) or 1 (disease)
    - confidence: probability of heart disease (0 to 1)
    - risk_level: Low / Medium / High
    """
    try:
        # Convert input to array in correct feature order
        input_data = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca,
            patient.thal
        ]])

        # Get prediction and probability
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        # Assign risk level based on probability
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log the request
        logging.info(f"Prediction: {prediction}, Confidence: {probability:.3f}, Risk: {risk_level}")

        return {
            "prediction": prediction,
            "confidence": round(probability, 3),
            "risk_level": risk_level,
            "message": "Heart disease detected" if prediction == 1 else "No heart disease detected"
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
