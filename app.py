# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st

# --- Custom CSS (only minimal background styling) ---
st.markdown("""
    <style>
        /* Light blue gradient background */
        body {
            background: linear-gradient(to bottom right, #a2d4f5, #e1f5fe);
            font-family: 'Segoe UI', sans-serif;
        }

        /* Button style */
        .stButton>button {
            background-color: #0288d1;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
        }

        .stButton>button:hover {
            background-color: #0277bd;
        }

        /* Result box style */
        .result-box {
            background-color: #e3f2fd;
            padding: 10px 18px;
            margin: 10px 0;
            border-left: 5px solid #0288d1;
            border-radius: 6px;
            font-size: 16px;
            color: #00334d;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Title and description
st.title("💧 Water Pollutants Predictor")
st.write("🔍 Predict the water pollutants based on **Year** and **Station ID**")

# User inputs
year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("🏷️ Enter Station ID", value='1')

# Prediction logic
if st.button('🔮 Predict'):
    if not station_id:
        st.warning('⚠️ Please enter the Station ID.')
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"📊 Predicted pollutant levels for Station ID '{station_id}' in {year_input}:")

        for p, val in zip(pollutants, predicted_pollutants):
            st.markdown(f"<div class='result-box'><b>{p}:</b> {val:.2f}</div>", unsafe_allow_html=True)
