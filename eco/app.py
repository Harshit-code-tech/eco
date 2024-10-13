import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib  # For loading the saved model

# Title of the app
st.title("FDI Impact on GDP Growth Prediction")

# Subtitle
st.write("This app predicts the impact of FDI inflows on GDP growth in India.")

# Load the trained model (replace 'your_model.pkl' with the actual model file)
model = joblib.load('random_forest_model.pkl')

# Input field for FDI inflows
fdi_inflows = st.number_input("Enter FDI inflows (in billion USD):", min_value=0.0)

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Convert input into a numpy array and reshape for prediction
    fdi_inflows = np.array(fdi_inflows).reshape(-1, 1)

    # Make the prediction
    gdp_growth_pred = model.predict(fdi_inflows)

    # Display the prediction
    st.write(f"Predicted GDP growth: {gdp_growth_pred[0]:.2f}%")
