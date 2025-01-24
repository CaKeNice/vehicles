

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Define the model training function
def train_model():
    # Load the dataset
    df = pd.read_csv('C:/Users/2019P/MLDP/vehicles.csv')

    # Preprocess the data as per the steps provided earlier
    df = df.drop(['id', 'url', 'region_url', 'image_url', 'description', 'lat', 'long', 'VIN', 'model', 'paint_color', 'region', 'state', 'county', 'posting_date', 'title_status', 'transmission'], axis=1)
    df.drop_duplicates()
    df = df[df['price'] < 1500000]
    threshold = 3

    missing_count = df.isna().sum(axis=1)
    df = df[missing_count < threshold]

    df['cylinders'] = df['cylinders'].str[:1]
    df['cylinders'] = pd.to_numeric(df['cylinders'], errors='coerce')
    df.loc[df['cylinders'].isna(), 'cylinders'] = df['cylinders'].mean()

    df.loc[df['odometer'].isna(), 'odometer'] = df['odometer'].mean()

    df.loc[df['fuel'].isna(), 'fuel'] = 'missing_fuel'
    df.loc[df['manufacturer'].isna(), 'manufacturer'] = 'missing_manufacturer'
    df.loc[df['condition'].isna(), 'condition'] = 'missing_condition'
    df.loc[df['drive'].isna(), 'drive'] = 'missing_drive'
    df.loc[df['size'].isna(), 'size'] = 'missing_size'
    df.loc[df['type'].isna(), 'type'] = 'missing_type'

    manufacturer_groups = {
        'acura': 'Japanese', 'audi': 'European', 'bmw': 'European', 'cadillac': 'American',
        'chevrolet': 'American', 'chrysler': 'American', 'dodge': 'American', 'ferrari': 'Luxury',
        'fiat': 'European', 'ford': 'American', 'gmc': 'American', 'honda': 'Japanese', 'hyundai': 'Korean',
        'infiniti': 'Japanese', 'jeep': 'American', 'kia': 'Korean', 'lexus': 'Japanese', 'lincoln': 'American',
        'mazda': 'Japanese', 'mercedes-benz': 'European', 'mercury': 'American', 'mini': 'European',
        'mitsubishi': 'Japanese', 'nissan': 'Japanese', 'porsche': 'Luxury', 'ram': 'American', 'rover': 'Luxury',
        'subaru': 'Japanese', 'tesla': 'American', 'toyota': 'Japanese', 'volkswagen': 'European', 'volvo': 'European'
    }

    # Define the grouping function
    def group_manufacturer(manufacturer):
        for key, group in manufacturer_groups.items():
            if key in manufacturer.lower():
                return group
        return 'Other'

    # Apply the function to create a new 'manufacturer_group' column
    df['manufacturer_group'] = df['manufacturer'].apply(group_manufacturer)
    df = df.drop(['manufacturer'], axis=1)

    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=['manufacturer_group', 'condition', 'fuel', 'drive', 'size', 'type'])

    X = df.drop('price', axis=1).to_numpy()
    y = df['price'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

    # Initialize and train the model
    model = DecisionTreeRegressor(
        max_depth=None,
        max_features=1.0,
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=4
    )

    model.fit(X_train, y_train)
    return model

# Train the model once at the beginning
DTR_model = train_model()

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
