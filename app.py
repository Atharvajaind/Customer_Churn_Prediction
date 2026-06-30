import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("models/churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn or not.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", value=70.0)
total = st.number_input("Total Charges", value=1000.0)

# Convert categorical values
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

if st.button("Predict"):
    data = pd.DataFrame([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        monthly,
        total
    ]], columns=[
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")