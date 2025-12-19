import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")

st.write("Enter car details to predict price")

# INPUTS


wheelbase = st.number_input("Wheelbase", value=98.76)   # default values can be adjusted
carlength = st.number_input("Car Length", value=174.05)
carwidth = st.number_input("Car Width", value=65.91)
carheight = st.number_input("Car Height", value=53.72)
curbweight = st.number_input("Curb Weight", value=2555.57)
enginesize = st.number_input("Engine Size", value=126.91)
boreratio = st.number_input("Bore Ratio", value=3.33)
stroke = st.number_input("Stroke", value=3.26)
compressionratio = st.number_input("Compression Ratio", value=10.14)
horsepower = st.number_input("Horsepower", value=104.12)

features = [wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower]

if st.button("Predict Price"):
    data = scaler.transform([features])
    prediction = model.predict(data)
    st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0])*10}")
