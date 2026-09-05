#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained model on test data and generates reports
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """Load trained model"""
    model_path = 'models/model_v1.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

def load_data():
    """Load test data"""
    df = pd.read_excel('data/raw/Delinquency_prediction_dataset.xlsx')
    return df

def evaluate_model():
    logger.info("MODEL EVALUATION")

    # Load model
    model = load_model()
    logger.info("Model loaded successfully")

    # Load data
    df = load_data()
    logger.info(f"Loaded {len(df)} rows")

    # Prepare features and target
    features = [
        'Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
        'Debt_to_Income_Ratio', 'Account_Tenure', 'Employment_Status',
        'Credit_Card_Type', 'Location'
    ]
    
    # Add engineered features if they exist in the model
    X = df[features].copy()
    y = df['Delinquent_Account'].copy()

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    metrics = {
        'auc_roc': roc_auc_score(y, y_proba),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'accuracy': accuracy_score(y, y_pred),
        'brier_score': brier_score_loss(y, y_proba)
    }

    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE METRICS:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Classification Report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y, y_pred, target_names=['No', 'Yes']))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Save metrics
    os.makedirs('models', exist_ok=True)
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("\nMetrics saved to models/evaluation_metrics.json")

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/roc_curve.png')
    plt.close()
    logger.info("ROC curve saved to models/roc_curve.png")

    return metrics

if __name__ == "__main__":
    evaluate_model()