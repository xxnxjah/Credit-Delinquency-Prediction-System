"""
API Tests for Credit Delinquency Prediction
Tests the FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api import app

# Create test client
client = TestClient(app)

# Sample test data
sample_data = {
    "Age": 45,
    "Income": 85000,
    "Credit_Score": 620,
    "Credit_Utilization": 0.75,
    "Missed_Payments": 4,
    "Debt_to_Income_Ratio": 0.42,
    "Account_Tenure": 8,
    "Employment_Status": "Unemployed",
    "Credit_Card_Type": "Platinum",
    "Location": "Houston",
    "Month_1": 1,
    "Month_2": 0,
    "Month_3": 2,
    "Month_4": 1,
    "Month_5": 0,
    "Month_6": 2
}

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()
    assert response.json()["service"] == "Credit Delinquency Prediction API"

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test the prediction endpoint with valid data"""
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "probability" in data
    assert "risk_score" in data
    assert "risk_class" in data
    assert "prediction" in data
    assert "timestamp" in data
    assert "model_version" in data
    
    # Check value ranges
    assert 0 <= data["probability"] <= 1
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_class"] in ["Low", "Medium", "High"]
    assert data["prediction"] in [0, 1]

def test_predict_endpoint_missing_field():
    """Test prediction with missing required field"""
    invalid_data = sample_data.copy()
    del invalid_data["Age"]  # Remove required field
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_endpoint_invalid_value():
    """Test prediction with invalid value"""
    invalid_data = sample_data.copy()
    invalid_data["Age"] = 150  # Invalid age (should be <= 100)
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity

def test_docs_endpoint():
    """Test that Swagger docs are accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_openapi_json():
    """Test that OpenAPI schema is accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/json"
    assert "openapi" in response.json()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])