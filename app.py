import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚚 Swiggy Delivery Time Prediction")

age = st.number_input("Age")
ratings = st.slider("Ratings", 1.0, 5.0)
weather = st.selectbox("Weather", [0,1,2,3])
traffic = st.selectbox("Traffic", [0,1,2,3])
vehicle_condition = st.selectbox("Vehicle Condition", [0,1,2])
type_of_order = st.selectbox("Type of Order", [0,1,2,3])
type_of_vehicle = st.selectbox("Vehicle Type", [0,1,2])
multiple_deliveries = st.number_input("Multiple Deliveries")
festival = st.selectbox("Festival", [0,1])
city_type = st.selectbox("City Type", [0,1,2])
is_weekend = st.selectbox("Weekend", [0,1])
pickup_time = st.number_input("Preparation Time")
order_hour = st.slider("Order Hour", 0, 23)
distance = st.number_input("Distance")

if st.button("Predict"):
    
    input_data = np.array([[
        age, ratings, weather, traffic, vehicle_condition,
        type_of_order, type_of_vehicle, multiple_deliveries,
        festival, city_type, is_weekend,
        pickup_time, order_hour, distance
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")