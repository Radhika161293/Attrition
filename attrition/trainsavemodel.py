# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Models
@st.cache_resource
def load_models():
    models = {}
    models['attrition_model'] = joblib.load('models/attrition_model.pkl')
    models['scaler'] = joblib.load('models/scaler.pkl')
    models['encoders'] = joblib.load('models/encoders.pkl')

    models['performance_model'] = joblib.load('models/performance_model.pkl')
    models['promotion_model'] = joblib.load('models/promotion_model.pkl')
    models['performance_scaler'] = joblib.load('models/performance_scaler.pkl')
    models['promotion_scaler'] = joblib.load('models/promotion_scaler.pkl')
    return models

models = load_models()

# ------------------ Streamlit App Layout ------------------

st.title("\U0001F3E2 Employee ML Prediction Dashboard")

task = st.sidebar.selectbox(
    "Select Prediction Task:",
    [
        "Attrition Prediction (Classification)",
        "Performance Rating Prediction (Logistic Regression)",
        "Promotion Likelihood Prediction (Regression)"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload Employee CSV File", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Sample:", df.head())
else:
    st.warning("\u26a0\ufe0f Please upload a dataset CSV to proceed!")

if uploaded_file is not None:

    if task == "Attrition Prediction (Classification)":
        st.header("\U0001F3AF Predicting Employee Attrition")

        model = models['attrition_model']
        scaler = models['scaler']
        encoders = models['encoders']

        # Preprocess uploaded file similar to training
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    st.error(f"Encoding error in column '{col}': {e}")

        X = df.drop('Attrition', axis=1, errors='ignore')

        expected_cols = getattr(scaler, 'feature_names_in_', X.columns)
        missing_cols = [col for col in expected_cols if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in expected_cols]

        if missing_cols:
            st.error(f"Missing columns required for prediction: {missing_cols}")
        elif extra_cols:
            st.warning(f"Ignoring extra columns: {extra_cols}")
            X = X[[col for col in expected_cols if col in X.columns]]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:,1]

        df['Predicted Attrition'] = y_pred
        df['Attrition Probability'] = y_proba
        st.write(df[['Predicted Attrition', 'Attrition Probability']])

        st.subheader("Attrition Probability Distribution")
        plt.figure(figsize=(6,4))
        sns.histplot(y_proba, bins=10, kde=True)
        plt.xlabel("Probability of Attrition")
        st.pyplot(plt.gcf())

    elif task == "Performance Rating Prediction (Logistic Regression)":
        st.header("\U0001F3AF Predicting Performance Rating")

        model = models['performance_model']
        scaler = models['performance_scaler']
        encoders = models['encoders']

        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    st.error(f"Encoding error in column '{col}': {e}")

        X = df
        expected_cols = getattr(scaler, 'feature_names_in_', X.columns)
        missing_cols = [col for col in expected_cols if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in expected_cols]

        if missing_cols:
            st.error(f"Missing columns required for prediction: {missing_cols}")
        elif extra_cols:
            st.warning(f"Ignoring extra columns: {extra_cols}")
            X = X[[col for col in expected_cols if col in X.columns]]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        df['Predicted Performance Rating'] = y_pred
        st.write(df[['Predicted Performance Rating']])

        st.subheader("Performance Rating Distribution")
        plt.figure(figsize=(6,4))
        sns.countplot(x=y_pred)
        plt.xlabel("Predicted Performance Rating")
        st.pyplot(plt.gcf())

    elif task == "Promotion Likelihood Prediction (Regression)":
        st.header("\U0001F3AF Predicting Years Since Last Promotion")

        model = models['promotion_model']
        scaler = models['promotion_scaler']
        encoders = models['encoders']

        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    st.error(f"Encoding error in column '{col}': {e}")

        X = df
        expected_cols = getattr(scaler, 'feature_names_in_', X.columns)
        missing_cols = [col for col in expected_cols if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in expected_cols]

        if missing_cols:
            st.error(f"Missing columns required for prediction: {missing_cols}")
        elif extra_cols:
            st.warning(f"Ignoring extra columns: {extra_cols}")
            X = X[[col for col in expected_cols if col in X.columns]]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        df['Predicted Years Since Last Promotion'] = y_pred
        st.write(df[['Predicted Years Since Last Promotion']])

        st.subheader("Years Since Last Promotion - Prediction vs Density")
        plt.figure(figsize=(6,4))
        sns.histplot(y_pred, kde=True)
        plt.xlabel("Predicted Years Since Last Promotion")
        st.pyplot(plt.gcf())

# ----------------- Footer ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
