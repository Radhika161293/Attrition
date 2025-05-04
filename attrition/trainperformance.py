# train_performance_model.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
df = pd.read_csv('employees.csv')

# Select features and target
features = [
    "Education", "JobInvolvement", "JobLevel",
    "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole"
]
target = "PerformanceRating"

df = df[features + [target]]

# Drop missing values
df.dropna(inplace=True)

# Train-test split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/performance_model.pkl')
joblib.dump(scaler, 'models/performance_scaler.pkl')

print("✅ Performance Model and Scaler saved successfully!")
