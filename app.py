import streamlit as st
import pickle
import numpy as np

# Load trained model (make sure you have saved it as model.pkl)
model = pickle.load(open("housing.pkl", "rb"))

# App title
st.title("🏠 House Price Prediction App")

st.write("Enter the details of the house below to predict its price:")

# Input fields
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2005)
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"])
garage = st.selectbox("Garage", ["Yes", "No"])

# Encode categorical variables (basic example — adjust to match your training dataset)
location_map = {"Urban": 2, "Suburban": 1, "Rural": 0}
condition_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
garage_map = {"Yes": 1, "No": 0}

location_val = location_map[location]
condition_val = condition_map[condition]
garage_val = garage_map[garage]

# Collect features into array
features = np.array([[area, bedrooms, bathrooms, floors, year_built,
                      location_val, condition_val, garage_val]])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(features)
    # Removed .2f formatting, now shows full raw value
    st.success(f"Predicted House Price: ${prediction[0]}")