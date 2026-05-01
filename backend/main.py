from fastapi import FastAPI
from pydantic import BaseModel
from predictor import predict_heart_disease

app = FastAPI(
    title="MedBuddy Heart Disease Prediction API",
    version="1.0",
    description="API for predicting heart disease risk based on patient data.",
)

# Input schema matching training features
class HeartDiseaseInput(BaseModel):
    age: int            
    sex: int            
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Heart Disease Prediction Endpoint
@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    # Convert Pydantic model to dictionary
    input_dict = input_data.model_dump()

    # Get prediction from model
    prediction, probability = predict_heart_disease(input_dict)

    return {
        "prediction": prediction,
        "probability": probability,
        "diagnosis": (
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease Detected"
        )
    }