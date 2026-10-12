
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

print("Loading California Housing dataset...")
data = fetch_california_housing()

print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(data.data, data.target)

print("Saving model files...")
joblib.dump(model, "house_model.pkl")
joblib.dump(data.feature_names, "feature_names.pkl")

print("SUCCESS: Both files have been saved!")
