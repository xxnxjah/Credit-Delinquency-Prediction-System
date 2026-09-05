import os
import logging
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading model...")
    try:
        model_path = os.getenv('MODEL_PATH', 'models/model_v1.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title="Credit Delinquency Prediction API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"
)

class CustomerInput(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Income: Optional[float] = Field(None, ge=0)
    Credit_Score: int = Field(..., ge=300, le=850)
    Credit_Utilization: float = Field(..., ge=0, le=1.0)
    Missed_Payments: int = Field(..., ge=0, le=12)
    Debt_to_Income_Ratio: float = Field(..., ge=0, le=1.0)
    Account_Tenure: int = Field(..., ge=0)
    Employment_Status: str
    Credit_Card_Type: str
    Location: str
    Month_1: int = Field(..., ge=0, le=2)
    Month_2: int = Field(..., ge=0, le=2)
    Month_3: int = Field(..., ge=0, le=2)
    Month_4: int = Field(..., ge=0, le=2)
    Month_5: int = Field(..., ge=0, le=2)
    Month_6: int = Field(..., ge=0, le=2)

class RiskResponse(BaseModel):
    probability: float
    risk_score: float
    risk_class: str
    prediction: int
    timestamp: str
    model_version: str

def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
    input_df['Avg_Payment_Status'] = input_df[month_cols].mean(axis=1)
    input_df['Recent_Missed_Flag'] = (input_df['Month_6'] == 2).astype(int)
    input_df['Utilization_Risk_Flag'] = (input_df['Credit_Utilization'] > 0.7).astype(int)
    return input_df

def predict_single(input_data: CustomerInput) -> Dict[str, Any]:
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = preprocess_input(input_df)
    
    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= 0.5)
    
    risk_score = round(prob * 100, 1)
    if prob >= 0.5:
        risk_class = "High"
    elif prob >= 0.3:
        risk_class = "Medium"
    else:
        risk_class = "Low"

    return {
        'probability': prob,
        'risk_score': risk_score,
        'risk_class': risk_class,
        'prediction': pred
    }

@app.get("/")
async def root():
    return {"service": "Credit Delinquency Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=RiskResponse)
async def predict(input_data: CustomerInput):
    try:
        result = predict_single(input_data)
        return RiskResponse(
            probability=result['probability'],
            risk_score=result['risk_score'],
            risk_class=result['risk_class'],
            prediction=result['prediction'],
            timestamp=datetime.now().isoformat(),
            model_version="v1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
