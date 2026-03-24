import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

st.set_page_config(page_title="Heart Disease Clinical Assessment", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'heart_disease_uci.csv')
    df = pd.read_csv(csv_path)
    df.drop(columns=['id', 'dataset'], inplace=True)
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    encoders = {}
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=['num'])
    y = df['num']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled, encoders, scaler

@st.cache_resource
def train_model(X_scaled, y):
    model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5, n_jobs=-1)
    model.fit(X_scaled, y)
    return model

X, y, X_scaled, encoders, scaler = load_and_preprocess_data()
rf_model = train_model(X_scaled, y)

st.title("Clinical Heart Disease Assessment Tool")
st.markdown("""
This diagnostic tool evaluates patient clinical records using a Random Forest algorithm trained on the UCI Heart Disease Dataset.
Please input the patient's physiological markers below to generate an automated severity classification.
""")
st.divider()

st.subheader("Patient Vitals & Clinical Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics & Symptoms**")
    age = st.number_input("Age (Years)", int(X['age'].min()), int(X['age'].max()), 50)
    sex = st.selectbox("Biological Sex", encoders['sex'].classes_)
    cp = st.selectbox("Chest Pain Classification", encoders['cp'].classes_)
    ca = st.number_input("Number of Major Vessels (0-3)", float(X['ca'].min()), float(X['ca'].max()), 0.0)

with col2:
    st.markdown("**Core Clinical Results**")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", float(X['trestbps'].min()), float(X['trestbps'].max()), 120.0)
    chol = st.number_input("Serum Cholesterol (mg/dl)", float(X['chol'].min()), float(X['chol'].max()), 200.0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", encoders['fbs'].classes_)
    thalch = st.number_input("Maximum Heart Rate Achieved", float(X['thalch'].min()), float(X['thalch'].max()), 150.0)

with col3:
    st.markdown("**ECG & Stress Test Data**")
    restecg = st.selectbox("Resting Electrocardiographic Results", encoders['restecg'].classes_)
    exang = st.selectbox("Exercise Induced Angina", encoders['exang'].classes_)
    oldpeak = st.number_input("ST Depression (Exercise vs Rest)", float(X['oldpeak'].min()), float(X['oldpeak'].max()), 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", encoders['slope'].classes_)
    thal = st.selectbox("Thalassemia Assessment", encoders['thal'].classes_)

st.divider()

user_data = {
    'age': age,
    'sex': encoders['sex'].transform([sex])[0],
    'cp': encoders['cp'].transform([cp])[0],
    'trestbps': trestbps,
    'chol': chol,
    'fbs': encoders['fbs'].transform([fbs])[0],
    'restecg': encoders['restecg'].transform([restecg])[0],
    'thalch': thalch,
    'exang': encoders['exang'].transform([exang])[0],
    'oldpeak': oldpeak,
    'slope': encoders['slope'].transform([slope])[0],
    'ca': ca,
    'thal': encoders['thal'].transform([thal])[0]
}

user_df = pd.DataFrame([user_data], columns=X.columns)
user_scaled = scaler.transform(user_df)

if st.button("Run Diagnostic Analysis", type="primary"):
    prediction = rf_model.predict(user_scaled)[0]
    
    st.subheader("Diagnostic Report")
    
    if prediction == 0:
        st.success("Result: The model indicates no presence of heart disease (Level 0).")
    else:
        st.warning(f"Result: The model indicates a potential presence of heart disease (Severity Level {prediction} out of 4). Clinical follow-up is recommended.")
        
    st.markdown("---")
    st.markdown("**Feature Importance Matrix (Technical Reference)**")
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    st.bar_chart(data=importance, x='Feature', y='Importance')
