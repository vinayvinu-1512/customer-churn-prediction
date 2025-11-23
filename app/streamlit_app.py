import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction App")
st.write("Predict whether a telecom customer will churn or not based on their details.")

# ----------------------- USER INPUT FIELDS -----------------------

gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])

PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])

Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# ----------------------- BUILD INPUT DATAFRAME -----------------------

df_input = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'PaperlessBilling': [PaperlessBilling],
    'Contract': [Contract],
    'PaymentMethod': [PaymentMethod],
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# ----------------------- PREDICT -----------------------

if st.button("Predict"):
    st.write("Input Summary:")
    st.write(df_input)

prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)[0][1]  # probability of staying

if prediction == 1:
    result = "Customer will stay"
else:
    result = "Customer will churn"

st.subheader("Prediction Result")
st.write(f"{result} — Probability: {prediction_proba * 100:.2f}%")



