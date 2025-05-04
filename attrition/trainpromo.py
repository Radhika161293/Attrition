# train_promotion_model.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('employees.csv')  # Make sure the path is correct

# 2. Feature and target selection
features = [
    "JobLevel", "TotalWorkingYears", "YearsInCurrentRole",
    "PerformanceRating", "Education"
]
target = "YearsSinceLastPromotion"

df = df[features + [target]]

# Drop missing values
df.dropna(inplace=True)

# 3. Split into features and target
X = df[features]
y = df[target]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Promotion Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/promotion_model.pkl')
joblib.dump(scaler, 'models/promotion_scaler.pkl')

print("✅ Promotion Model and Scaler saved successfully in 'models/' folder!")
