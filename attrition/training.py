# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from preprocessing import load_data, preprocess_data

# 1. Load and preprocess the dataset
filepath = 'employees.csv'
df = load_data(filepath)
df, encoders = preprocess_data(df)

# 2. Define features and target
target_column = 'attrition'  # <-- make sure this matches your dataset (already standardized lowercase)
X = df.drop(target_column, axis=1)
y = df[target_column]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# 7. Save the model and scaler
joblib.dump(model, 'models/attrition_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl') 
print("Model and scaler saved successfully!")
