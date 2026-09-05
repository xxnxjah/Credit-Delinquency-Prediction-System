import joblib
import pandas as pd

#Load model
model=joblib.load('models/model_v1.pkl')

#Sample customer data
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

#Predict
sample_df = pd.DataFrame([sample])
prob = model.predict_proba(sample_df)[0, 1]
pred = int(prob >= 0.5)

risk_score = round(prob * 100, 1)
if prob >= 0.5:
    risk_class = "High"
elif prob >= 0.3:
    risk_class = "Medium"
else:
    risk_class = "Low"

print("RISK ASSESSMENT RESULT")
print(f"Probability of Delinquency: {prob:.2%}")
print(f"Risk Score: {risk_score}")
print(f"Risk Class: {risk_class}")
print(f"Prediction: {'Delinquent' if pred == 1 else 'Not Delinquent'}")
