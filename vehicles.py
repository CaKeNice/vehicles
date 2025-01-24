import streamlit as st
import pandas as pd
import numpy as np
from sklearn.externals import joblib

DTR_model = joblib.load('model_dtr.joblib')
# Add a title
st.title("Vehicle Price Prediction")

# Input form for vehicle details
st.subheader("Enter the vehicle details")
# User inputs for the vehicle attributes
year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
cylinders = st.selectbox("Cylinders", options=[1, 2, 3, 4, 5, 6, 7, 8])
odometer = st.number_input("Odometer (in miles)", min_value=0)
fuel = st.selectbox("Fuel type", options=['gas', 'diesel', 'electric', 'hybrid', 'other'])
manufacturer = st.selectbox("Manufacturer", options=['japanese','european','american','luxury','korean','other'])
condition = st.selectbox("Condition", options=['others', 'new', 'excellent', 'good', 'fair', 'salvage'])
drive = st.selectbox("Drive", options=['others', '4wd', 'fwd', 'rwd'])
size = st.selectbox("Size", options=['others', 'compact', 'full-size', 'midsize'])
type_ = st.selectbox("Vehicle Type", options=['others', 'sedan', 'suv', 'truck', 'wagon', 'coupe', 'van'])

# Predict button
if st.button("Predict Price"):
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
    'year': [year],
    'cylinders': [cylinders],
    'odometer': [odometer],
    
    # Fuel type features
    'fuel_missing_fuel': [1 if fuel == 'missing_fuel' else 0],
    'fuel_gas': [1 if fuel == 'gas' else 0],
    'fuel_diesel': [1 if fuel == 'diesel' else 0],
    'fuel_electric': [1 if fuel == 'electric' else 0],
    'fuel_hybrid': [1 if fuel == 'hybrid' else 0],
    'fuel_other': [1 if fuel == 'other' else 0],
    
    # Manufacturer group features
    'manufacturer_group_Japanese': [1 if manufacturer =='japanese'else 0],
    'manufacturer_group_European': [1 if manufacturer =='european'else 0],
    'manufacturer_group_American':[1 if manufacturer =='american'else 0],
    'manufacturer_group_Luxury': [1 if manufacturer =='luxury'else 0],
    'manufacturer_group_Korean': [1 if manufacturer =='korean'else 0],
    'manufacturer_group_Other': [1 if manufacturer =='other'else 0],
    
    # Condition features
    'condition_missing_condition': [1 if condition == 'others' else 0],
    'condition_new': [1 if condition == 'new' else 0],
    'condition_excellent': [1 if condition == 'excellent' else 0],
    'condition_good': [1 if condition == 'good' else 0],
    'condition_fair': [1 if condition == 'fair' else 0],
    'condition_salvage': [1 if condition == 'salvage' else 0],
    
    # Drive features
    'drive_missing_drive': [1 if drive == 'others' else 0],
    'drive_4wd': [1 if drive == '4wd' else 0],
    'drive_fwd': [1 if drive == 'fwd' else 0],
    'drive_rwd': [1 if drive == 'rwd' else 0],
    
    # Size features
    'size_missing_size': [1 if size == 'others' else 0],
    'size_compact': [1 if size == 'compact' else 0],
    'size_full-size': [1 if size == 'full-size' else 0],
    'size_midsize': [1 if size == 'midsize' else 0],  # Changed from 'size_mid-size'
    'size_sub-compact': [1 if size == 'sub-compact' else 0],  # Added missing size
    
    # Vehicle type features
    'type_missing_type': [1 if type_ == 'others' else 0],
    'type_sedan': [1 if type_ == 'sedan' else 0],
    'type_suv': [1 if type_ == 'suv' else 0],
    'type_truck': [1 if type_ == 'truck' else 0],
    'type_wagon': [1 if type_ == 'wagon' else 0],
    'type_coupe': [1 if type_ == 'coupe' else 0],
    'type_van': [1 if type_ == 'van' else 0],
    'type_convertible': [0],  # Added missing vehicle type
    'type_pickup': [0],  # Added missing vehicle type
    'type_hatchback': [0],  # Added missing vehicle type
    'type_offroad': [0],  # Added missing vehicle type
    'type_SUV': [0],  # Added missing vehicle type
    'type_mini-van': [0],  # Added missing vehicle type
    'type_bus': [0],  # Added missing vehicle type
    'type_other': [0],  # Added missing vehicle type
    
    # Add price column as 0 (dummy) since it is the target in your mode
})

    # Make prediction using the trained model
    prediction = DTR_model.predict(user_input)

    # Display the predicted price
    st.write(f"Predicted vehicle price: ${prediction[0]:,.2f}")
