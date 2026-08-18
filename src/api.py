from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Swiggy Delivery ETA Prediction API",
    description="API for predicting food delivery time",
    version="1.0.0"
)


# Load trained model
model = joblib.load(
    "models/xgboost_eta_model.joblib"
)

# Load preprocessing pipeline
preprocessor = joblib.load(
    "models/eta_preprocessor.joblib"
)


# Input schema
class DeliveryInput(BaseModel):
    Distance_km: float
    order_hour: float
    day_of_week: int
    month: int
    is_weekend: int
    is_peak_hour: int
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    Vehicle_condition: int
    multiple_deliveries: int
    Weatherconditions: str
    Road_traffic_density: str
    Type_of_order: str
    Type_of_vehicle: str
    Festival: str
    City: str


@app.get("/")
def home():
    return {
        "message": "Swiggy ETA Prediction API is running"
    }


@app.post("/predict")
def predict(data: DeliveryInput):

    # Convert request to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Apply the same preprocessing used during training
    processed_data = preprocessor.transform(
        input_data
    )

    # Make prediction
    prediction = model.predict(
        processed_data
    )[0]

    # Prevent unrealistic negative values
    prediction = max(0, prediction)

    return {
        "predicted_delivery_time_minutes": round(
            float(prediction),
            2
        )
    }