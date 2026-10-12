
import os
import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.write("Estimate a California district's median house value using machine learning.")

MODEL_FILE = "house_model.pkl"
FEATURES_FILE = "feature_names.pkl"

@st.cache_resource
def load_model():
    housing = fetch_california_housing(download_if_missing=True)

    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_FILE):
        model = joblib.load(MODEL_FILE)
        feature_names = joblib.load(FEATURES_FILE)
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(housing.data, housing.target)
        feature_names = list(housing.feature_names)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(feature_names, FEATURES_FILE)

    return model, feature_names

try:
    with st.spinner("Loading dataset and preparing model..."):
        model, feature_names = load_model()
except Exception as e:
    st.error("The model could not be prepared. Check the app logs for details.")
    st.exception(e)
    st.stop()

labels = {
    "MedInc": "Median income (tens of thousands of dollars)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms",
    "AveBedrms": "Average bedrooms",
    "Population": "Population",
    "AveOccup": "Average household occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

housing = fetch_california_housing()
defaults = [3.5, 28.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -119.0]

with st.form("prediction_form"):
    st.subheader("Enter housing details")
    values = {}

    for feature, default in zip(feature_names, defaults):
        values[feature] = st.number_input(
            labels.get(feature, feature),
            value=float(default)
        )

    submitted = st.form_submit_button("Predict House Value")

if submitted:
    input_df = pd.DataFrame([values], columns=feature_names)
    prediction = float(model.predict(input_df)[0]) * 100000

    st.success(f"Estimated median value: ${prediction:,.2f}")
    st.caption(
        "This estimate uses the California Housing dataset. "
        "It is not a current valuation of an individual property."
    )
