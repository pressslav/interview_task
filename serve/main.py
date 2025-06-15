from fastapi import FastAPI
import joblib
import pandas as pd
import json
from pydantic import BaseModel
import os
from typing import Optional
import logging
import time
from starlette.requests import Request

#in order to log the latency and metrics 
#I'm creating a file in the 'serve' directory inside the container
LOG_FILE = os.path.join(os.path.dirname(__file__), "app.log")

#Configure logging to write to a file and to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#I'm defining the prediction serving app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API to predict customer churn using a pre-trained model.",
    version="1.0"
)

#I'm logging the latency of each incoming request in ms and rounding it to 2 decimal places for legibility
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # in milliseconds
    logger.info(f"request_latency_ms={process_time:.2f} path={request.url.path}")
    return response

#I'm loading the model and feature list from the models directory
MODEL_DIR = "/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        model_features = json.load(f)
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model or features: {e}")
    model = None
    model_features = []

#I'm defining the structure and data types for the input JSON.
#Based on the columns from the telecom_customer_churn.csv dataset, excluding identifiers.

#this is not an optimal configuration, I would have preferred to import the MLflowClient module
#and use the model_uri to load the model and feature list instead but I got some errors with that approach 

class ChurnInput(BaseModel):
    Gender: Optional[str] = None
    Age: Optional[int] = None
    Married: Optional[str] = None
    Number_of_Dependents: Optional[int] = None
    City: Optional[str] = None
    Zip_Code: Optional[int] = None
    Latitude: Optional[float] = None
    Longitude: Optional[float] = None
    Number_of_Referrals: Optional[int] = None
    Tenure_in_Months: Optional[int] = None
    Offer: Optional[str] = None
    Phone_Service: Optional[str] = None
    Avg_Monthly_Long_Distance_Charges: Optional[float] = None
    Multiple_Lines: Optional[str] = None
    Internet_Service: Optional[str] = None
    Internet_Type: Optional[str] = None
    Avg_Monthly_GB_Download: Optional[int] = None
    Online_Security: Optional[str] = None
    Online_Backup: Optional[str] = None
    Device_Protection_Plan: Optional[str] = None
    Premium_Tech_Support: Optional[str] = None
    Streaming_TV: Optional[str] = None
    Streaming_Movies: Optional[str] = None
    Streaming_Music: Optional[str] = None
    Unlimited_Data: Optional[str] = None
    Contract: Optional[str] = None
    Paperless_Billing: Optional[str] = None
    Payment_Method: Optional[str] = None
    Monthly_Charge: Optional[float] = None
    Total_Charges: Optional[float] = None
    Total_Refunds: Optional[float] = None
    Total_Extra_Data_Charges: Optional[int] = None
    Total_Long_Distance_Charges: Optional[float] = None
    Total_Revenue: Optional[float] = None

#I'm exposing the prediction endpoint to take customer data as input and return a churn prediction
@app.post("/predict")
async def predict(input_data: ChurnInput):
    if not model or not model_features:
        return {"error": "Model not loaded. Please check server logs."}

    #Convert Pydantic model to a dictionary, filtering out None values
    input_dict = {k: v for k, v in input_data.dict().items() if v is not None}
    
    if not input_dict:
        return {"error": "No input data provided."}
        
    input_df = pd.DataFrame([input_dict])

    #I'm one-hot encoding the categorical features to ensure the input matches 
    #the format the model was trained on
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    #I'm aligning the columns with the model's training features
    # and creating a new DataFrame with a single row of zeros and the model's features
    final_input = pd.DataFrame(0, index=[0], columns=model_features)
    
    #Update the DataFrame with the one-hot encoded values from the input
    #This ensures that only the columns present in both are updated
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].values

    #Make prediction
    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[:, 1] # Probability of churn

    prediction_label = "Churn" if prediction[0] == 1 else "Stay"
    logger.info(f"prediction_result={prediction_label}")

    return {
        "prediction": prediction_label,
        "probability": float(probability[0])
    }

@app.get("/")
def read_root():
    return {"message": "this is an api for predictions."} 