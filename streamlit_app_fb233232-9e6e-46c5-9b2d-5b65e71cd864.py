import streamlit as st
import pandas as pd
import pickle

# Load the saved model
model_path = "models/random_forest_regressor.pkl"  # Update this to the best model's file name
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.title("Crop Production Prediction")
st.write("This application predicts crop production (in tons) based on agricultural factors like area harvested (in hectares) and yield (in kg/ha).")

# Sidebar for user input
st.sidebar.header("Input Features")
area_harvested = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0, value=1000.0, step=1.0)
yield_value = st.sidebar.number_input("Yield (kg/ha)", min_value=0.0, value=1000.0, step=1.0)

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = model.predict([[area_harvested, yield_value]])
    st.write(f"### Predicted Crop Production: {prediction[0]:.2f} tons")

# Additional insights
st.write("### About the Model")
st.write("The model used for this prediction is a Random Forest Regressor, trained on historical agricultural data.")
st.write("You can adjust the input features in the sidebar to see how they affect the predicted production.")
