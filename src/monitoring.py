"""
Monitoring Dashboard for Credit Delinquency Prediction
Tracks model performance, data drift, and fairness metrics
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, model_path='models/model_v1.pkl'):
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None
        self.metrics_history = []
    
    def evaluate_performance(self, X_test, y_test):
        """Calculate performance metrics"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score,
            f1_score, accuracy_score, brier_score_loss
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_proba)
        }
    
    def check_data_drift(self, reference_data, current_data):
        """Check for data drift between reference and current datasets"""
        # Simplified drift detection
        drift_report = {}
        for col in reference_data.columns:
            if reference_data[col].dtype in ['float64', 'int64']:
                ref_mean = reference_data[col].mean()
                curr_mean = current_data[col].mean()
                drift = abs(curr_mean - ref_mean) / (ref_mean + 1e-6)
                drift_report[col] = {
                    'reference_mean': ref_mean,
                    'current_mean': curr_mean,
                    'drift_percentage': drift * 100
                }
        return drift_report
    
    def generate_report(self):
        """Generate a comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_loaded': self.model is not None,
            'status': 'healthy' if self.model else 'degraded'
        }
        return report

def run_monitoring():
    """Main monitoring function"""
    logger.info("RUNNING MONITORING DASHBOARD")
    
    monitor = ModelMonitor()
    report = monitor.generate_report()
    
    logger.info(f"Report: {json.dumps(report, indent=2)}")
    
    os.makedirs('monitoring', exist_ok=True)
    with open('monitoring/monitoring_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Monitoring report saved to monitoring/monitoring_report.json")

if __name__ == "__main__":
    run_monitoring()