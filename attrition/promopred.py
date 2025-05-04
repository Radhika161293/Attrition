import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\Radhika\Desktop\project\projects\attrition\employees.csv")

# Print available columns (optional debug step)
print("Columns in dataset:", df.columns.tolist())

# --- Step 1: Clean and standardize column names ---
df.columns = df.columns.str.strip()

# --- Step 2: Feature and target selection ---
features = [
    "JobLevel", "TotalWorkingYears", "YearsInCurrentRole",
    "PerformanceRating", "Education"
]
target = "YearsSinceLastPromotion"

# Filter dataset
df = df[features + [target]]

# Drop missing values
df.dropna(inplace=True)

# Encode categorical features if needed
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Step 3: Split data ---
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train Random Forest Regressor ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Predict and Evaluate ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n📊 Regression Evaluation Metrics for Promotion Prediction:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")




# --- Step 6: Plot Actual vs Predicted ---
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Years Since Last Promotion")
plt.ylabel("Predicted Years Since Last Promotion")
plt.title("Actual vs Predicted - Years Since Last Promotion")
plt.tight_layout()
plt.savefig("promotion_likelihood_prediction.png")
plt.show()
