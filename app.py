import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Car Price Prediction App")

st.write("Enter key car details to predict price. The model uses additional internal features for accuracy.")

# Numerical inputs (10 key features)
st.header("Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    wheelbase = st.number_input("Wheelbase", value=98.76, min_value=80.0, max_value=120.0)
    carlength = st.number_input("Car Length", value=174.05, min_value=140.0, max_value=210.0)
    carwidth = st.number_input("Car Width", value=65.91, min_value=60.0, max_value=75.0)
    carheight = st.number_input("Car Height", value=53.72, min_value=45.0, max_value=65.0)

with col2:
    curbweight = st.number_input("Curb Weight", value=2555.57, min_value=1500.0, max_value=4000.0)
    enginesize = st.number_input("Engine Size", value=126.91, min_value=60.0, max_value=350.0)
    boreratio = st.number_input("Bore Ratio", value=3.33, min_value=2.5, max_value=4.0)
    stroke = st.number_input("Stroke", value=3.26, min_value=2.0, max_value=5.0)

with col3:
    compressionratio = st.number_input("Compression Ratio", value=10.14, min_value=7.0, max_value=25.0)
    horsepower = st.number_input("Horsepower", value=104.12, min_value=50.0, max_value=300.0)

# No categorical inputs shown - using defaults internally

if st.button("Predict Price"):
    # Create feature vector with defaults
    features = np.zeros(79)
    
    # Shown numerical features (apply log to skewed ones)
    features[0] = wheelbase
    features[1] = carlength
    features[2] = carwidth
    features[3] = carheight
    features[4] = curbweight
    features[5] = np.log1p(enginesize)  # skewed
    features[6] = boreratio
    features[7] = stroke
    features[8] = np.log1p(compressionratio)  # skewed
    features[9] = np.log1p(horsepower)  # skewed
    
    # Hidden numerical features (with log for skewed)
    features[10] = np.log1p(5125.37)  # peakrpm
    features[11] = np.log1p(25.22)    # citympg
    features[12] = np.log1p(30.75)    # highwaympg
    
    # Categorical defaults (most common)
    features[36] = 1  # CarName_toyota
    features[42] = 1  # fueltype_gas
    features[43] = 1  # aspiration_std
    features[45] = 1  # doornumber_four
    features[50] = 1  # carbody_sedan
    features[53] = 1  # drivewheel_fwd
    features[55] = 1  # enginelocation_front
    features[59] = 1  # enginetype_ohc
    features[63] = 1  # cylindernumber_four
    features[69] = 1  # fuelsystem_mpfi
    
    # Scale and predict
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0])}")
