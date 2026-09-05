"""
Unit Tests for Credit Delinquency Prediction Model
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def test_model_exists():
    """Test that model file exists"""
    assert os.path.exists('models/model_v1.pkl'), "Model file not found"

def test_model_loads():
    """Test that model loads successfully"""
    model = joblib.load('models/model_v1.pkl')
    assert model is not None

def test_prediction():
    """Test that model makes predictions"""
    model = joblib.load('models/model_v1.pkl')
    
    sample = {
        'Age': 45,
        'Income': 85000,
        'Credit_Score': 620,
        'Credit_Utilization': 0.75,
        'Missed_Payments': 4,
        'Debt_to_Income_Ratio': 0.42,
        'Account_Tenure': 8,
        'Employment_Status': 'Unemployed',
        'Credit_Card_Type': 'Platinum',
        'Location': 'Houston',
        'Avg_Payment_Status': 1.2,
        'Recent_Missed_Flag': 1,
        'Utilization_Risk_Flag': 1
    }
    
    sample_df = pd.DataFrame([sample])
    prob = model.predict_proba(sample_df)[0, 1]
    
    assert 0 <= prob <= 1, "Probability should be between 0 and 1"

def test_data_loading():
    """Test that data loads correctly"""
    df = pd.read_excel('data/raw/Delinquency_prediction_dataset.xlsx')
    assert len(df) > 0, "Data should have at least one row"
    assert 'Delinquent_Account' in df.columns, "Target column missing"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])