import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\Radhika\Desktop\project\projects\attrition\employees.csv")

# Print available columns (optional debug step)
print("Columns in dataset:", df.columns.tolist())

# --- Step 1: Clean and standardize column names ---
df.columns = df.columns.str.strip()

# --- Step 2: Feature and target selection ---
features = [
    "Education", "JobInvolvement", "JobLevel",
    "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole"
]
target = "PerformanceRating"

# Filter dataset
df = df[features + [target]]

# Check for missing values
df.dropna(inplace=True)

# Encode categorical features (if any)
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Step 3: Split data ---
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train model ---
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Evaluate model ---
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n📊 Classification Evaluation Metrics:")
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Step 6: Confusion Matrix Plot ---
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Performance Rating Prediction")
plt.tight_layout()
plt.savefig("performance_rating_classification.png")
plt.show()
