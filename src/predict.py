# src/predict.py
import joblib
import pandas as pd
import numpy as np

def load_model(path='models/churn_model.pkl'):
    return joblib.load(path)

def predict_single(model, input_dict):
    # input_dict is a mapping feature -> value (strings or numbers)
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0,1]
    return int(pred), float(prob)

if __name__ == "__main__":
    model = load_model()
    # example: fill with appropriate features found in your dataset
    sample = {
        "gender":"Female",
        "SeniorCitizen":0,
        "Partner":"Yes",
        "Dependents":"No",
        "tenure":12,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"DSL",
        "OnlineSecurity":"No",
        "OnlineBackup":"Yes",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 800.5
    }
    p, pr = predict_single(model, sample)
    print("Predicted churn:", p, "prob:", pr)
