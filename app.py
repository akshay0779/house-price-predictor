
import os
import streamlit as st
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "house_model.pkl"
FEATURES_FILE = "feature_names.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    with st.spinner("Preparing the model for the first time..."):
        housing = fetch_california_housing()

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(housing.data, housing.target)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(list(housing.feature_names), FEATURES_FILE)

model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURES_FILE)
