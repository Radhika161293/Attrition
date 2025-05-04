import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\Radhika\Desktop\project\projects\attrition\employees.csv")

# --- Step 1: Standardize column names for internal use ---
df.columns = df.columns.str.strip()  # Strip whitespace but preserve original case

# --- Step 2: Select relevant columns based on your list ---
features = [
    "Age", "Department", "MonthlyIncome", "JobSatisfaction",
    "YearsAtCompany", "MaritalStatus", "OverTime"
]
target = "Attrition"

df = df[features + [target]]

# --- Step 3: Encode categorical features ---
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Step 4: Encode target if necessary ---
if df[target].dtype == "object":
    df[target] = LabelEncoder().fit_transform(df[target])

# --- Step 5: Split data ---
X = df[features]
y = df[target]
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)  # Ensure target is numeric

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: Train Decision Tree Model ---
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# --- Step 6: Predictions and Evaluation ---
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# --- Step 7: Evaluation Metrics ---
print("📊 Evaluation Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Step 8: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.clf()

# --- Step 9: ROC Curve ---
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.clf()

print("✅ Model evaluation completed. Confusion matrix and ROC curve saved as images.")

