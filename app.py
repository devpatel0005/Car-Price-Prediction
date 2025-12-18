import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")

st.write("Enter car details to predict price")

# INPUTS (adjust count if needed)
features = []
for i in range(10):   # keep same feature count used during training
    features.append(st.number_input(f"Feature {i+1}", value=0.0))

if st.button("Predict Price"):
    data = scaler.transform([features])
    prediction = model.predict(data)
    st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0])}")
