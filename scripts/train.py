#!/usr/bin/env python3
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, accuracy_score, brier_score_loss,
    confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("CREDIT DELINQUENCY PREDICTION - TRAINING PIPELINE")
    
    data_path = 'data/raw/Delinquency_prediction_dataset.xlsx'
    if not os.path.exists(data_path):
        data_path = '../data/raw/Delinquency_prediction_dataset.xlsx'
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_excel(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Clean data
    status_map = {'EMP': 'Employed', 'employed': 'Employed', 'Employed': 'Employed',
                  'Self-employed': 'Self-employed', 'Unemployed': 'Unemployed', 'retired': 'Retired'}
    df['Employment_Status'] = df['Employment_Status'].map(status_map).fillna('Employed')
    df['Credit_Utilization'] = df['Credit_Utilization'].clip(upper=1.0)
    
    payment_map = {'On-time': 0, 'Late': 1, 'Missed': 2}
    month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
    for col in month_cols:
        df[col] = df[col].map(payment_map).fillna(0)
    
    df['Avg_Payment_Status'] = df[month_cols].mean(axis=1)
    df['Recent_Missed_Flag'] = (df['Month_6'] == 2).astype(int)
    df['Utilization_Risk_Flag'] = (df['Credit_Utilization'] > 0.7).astype(int)
    
    features = ['Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
                'Debt_to_Income_Ratio', 'Account_Tenure', 'Employment_Status',
                'Credit_Card_Type', 'Location', 'Avg_Payment_Status',
                'Recent_Missed_Flag', 'Utilization_Risk_Flag']
    
    X = df[features].copy()
    y = df['Delinquent_Account'].copy()
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    num_features = ['Age', 'Income', 'Credit_Score', 'Credit_Utilization', 
                    'Missed_Payments', 'Debt_to_Income_Ratio', 'Account_Tenure',
                    'Avg_Payment_Status', 'Recent_Missed_Flag', 'Utilization_Risk_Flag']
    cat_features = ['Employment_Status', 'Credit_Card_Type', 'Location']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
    ])
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=scale_pos_weight,
                                   eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    pipeline = ImbPipeline([('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
                            ('classifier', xgb_model)])
    
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'brier_score': brier_score_loss(y_test, y_proba)
    }
    
    logger.info("TEST SET METRICS:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("="*60)
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/model_v1.pkl')
    logger.info("Model saved to models/model_v1.pkl")
    
    with open('models/metrics_v1.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Feature importance
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_feature_names)
    
    importance = pipeline.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False)
    feat_imp.to_csv('models/feature_importance_v1.csv', index=False)
    
    logger.info("\nTop 10 Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    logger.info("All artifacts saved to models/")
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
