# preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

def clean_column_names(df):
    """
    Standardizes column names: lowercase, underscores instead of spaces, remove special characters.
    """
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace(r'[^\w\s]', '', regex=True)
    )
    return df
def remove_outliers_iqr(df, columns):
    """
    Removes outliers from specified columns using the IQR method.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df


def load_data(filepath):
    df = pd.read_csv("employees.csv")
    df = clean_column_names(df)  # <- Standardize column names immediately after loading
    return df

def preprocess_data(df):
    """
    Preprocesses the dataframe: handles missing values and encodes categorical variables.
    """
    # Handle missing values (basic dropna for now)
    df = df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders
