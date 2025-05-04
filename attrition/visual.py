# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocessing import load_data, preprocess_data

# Load and preprocess data
df = load_data('employees.csv')
df, encoders = preprocess_data(df)

# Create folder for saving EDA images
os.makedirs("eda_outputs", exist_ok=True)

# Overview
print("🔍 Shape:", df.shape)
print("🧾 Columns:", df.columns.tolist())
print("📊 Attrition Class Distribution:\n", df['attrition'].value_counts())

# 1. Class Distribution Plot
sns.countplot(data=df, x='attrition')
plt.title('Attrition Distribution')
plt.savefig("eda_outputs/attrition_distribution.png")
plt.clf()

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig("eda_outputs/correlation_matrix.png")
plt.clf()

# 3. Boxplots to see outliers in numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x='attrition', y=col, data=df)
    plt.title(f'{col} vs Attrition')
    plt.savefig(f"eda_outputs/boxplot_{col}.png")
    plt.clf()

# 4. Categorical vs Attrition
# categorical_cols = df.select_dtypes(include='int64').columns.difference([col for col in numeric_cols if df[col].n_])
