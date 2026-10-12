
import streamlit as st
import joblib
import numpy as np

# Load saved model and feature names
model = joblib.load("house_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

st.title("🏡 House Price Predictor")
st.write("Enter the housing details to estimate the median house value.")

# Dataset feature ranges are approximate and intended for demonstration.
defaults = [3.5, 28.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -119.0]

labels = {
    "MedInc": "Median income (in tens of thousands of dollars)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

values = []

with st.form("house_form"):
    for i, name in enumerate(feature_names):
        values.append(
            st.number_input(
                labels.get(name, name),
                value=float(defaults[i]),
                format="%.4f"
            )
        )

    submitted = st.form_submit_button("Predict House Value")

if submitted:
    prediction = model.predict(np.array(values).reshape(1, -1))[0]
    price = prediction * 100000

    st.success(f"Estimated median house value: ${price:,.2f}")
    st.caption(
        "Demonstration estimate based on historical California Housing data, "
        "not a current property valuation."
    )
