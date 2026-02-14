import streamlit as st
import pickle
import pandas as pd
import numpy as np
import uuid
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load saved files
model = pickle.load(open("house_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
training_columns = pickle.load(open("training_columns.pkl", "rb"))

# Title
st.title("🏠 Hyderabad House Price Prediction")
st.markdown("### Enter House Details Below")

# Sidebar Inputs
st.sidebar.header("House Features")

area = st.sidebar.number_input(
    "Area (in sqft)",
    min_value=100,
    max_value=10000,
    value=1200,
    key="area_input"
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2,
    key="bedroom_input"
)

washrooms = st.sidebar.number_input(
    "Number of Washrooms",
    min_value=1,
    max_value=10,
    value=2,
    key="washroom_input"
)

# Create input dataframe
input_data = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Washrooms": [washrooms]
})

# Convert to dummies (same as training)
input_data = pd.get_dummies(input_data)

# Match training columns
input_data = input_data.reindex(columns=training_columns, fill_value=0)

# Scale input
scaled_data = scaler.transform(input_data)

# Predict button
if st.button("Predict Price 💰"):

    prediction = model.predict(scaled_data)

    # Generate Unique Prediction ID
    prediction_id = "HYD-" + datetime.now().strftime("%Y%m%d") + "-" + str(uuid.uuid4())[:6]

    st.success(f"Prediction ID: {prediction_id}")
    st.success(f"Estimated House Price: ₹ {round(prediction[0],2):,}")

    st.info("Prediction generated successfully.")

# Footer
st.markdown("---")
st.caption("Mini Project | Machine Learning | Streamlit App")
