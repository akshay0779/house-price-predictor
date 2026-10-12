
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("LINEAR REGRESSION")
print("R2 Score:", round(r2_score(y_test, lr_predictions), 3))
print("MSE:", round(mean_squared_error(y_test, lr_predictions), 3))

# 4. Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\nRANDOM FOREST")
print("R2 Score:", round(r2_score(y_test, rf_predictions), 3))
print("MSE:", round(mean_squared_error(y_test, rf_predictions), 3))

# 5. Display example prediction
print("\nEXAMPLE PREDICTION")
print("Actual price: $", round(y_test[0] * 100000, 2))
print("Linear Regression: $", round(lr_predictions[0] * 100000, 2))
print("Random Forest: $", round(rf_predictions[0] * 100000, 2))
