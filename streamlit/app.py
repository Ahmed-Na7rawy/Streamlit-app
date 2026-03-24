import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'heart_disease_uci.csv')
    df = pd.read_csv(csv_path)
    df.drop(columns=['id', 'dataset'], inplace=True)
    # Fill NAs
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Keep encoders to transform user string input back to integers
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
    # Use best known parameters
    model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5, n_jobs=-1)
    model.fit(X_scaled, y)
    return model

# 1. Load Data
X, y, X_scaled, encoders, scaler = load_and_preprocess_data()
# 2. Load Model
rf_model = train_model(X_scaled, y)

# ================= APP UI =================
st.title("Heart Disease Severity Predictor ❤️🩺")
st.markdown("""
This web application uses a **Random Forest Classifier** trained on the UCI Heart Disease Dataset. 
Use the **sidebar** on the left to input patient details and receive an immediate severity classification (0-4).
""")

# ================= SIDEBAR =================
st.sidebar.header("Patient Vitals Input")

age = st.sidebar.slider("Age", int(X['age'].min()), int(X['age'].max()), 50)
sex = st.sidebar.selectbox("Sex", encoders['sex'].classes_)
cp = st.sidebar.selectbox("Chest Pain Type (cp)", encoders['cp'].classes_)
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", float(X['trestbps'].min()), float(X['trestbps'].max()), 120.0)
chol = st.sidebar.slider("Cholesterol (chol)", float(X['chol'].min()), float(X['chol'].max()), 200.0)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", encoders['fbs'].classes_)
restecg = st.sidebar.selectbox("Resting ECG (restecg)", encoders['restecg'].classes_)
thalch = st.sidebar.slider("Max Heart Rate Achieved (thalch)", float(X['thalch'].min()), float(X['thalch'].max()), 150.0)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", encoders['exang'].classes_)
oldpeak = st.sidebar.slider("ST depression induced by exercise (oldpeak)", float(X['oldpeak'].min()), float(X['oldpeak'].max()), 1.0)
slope = st.sidebar.selectbox("Peak exercise ST segment slope (slope)", encoders['slope'].classes_)
ca = st.sidebar.slider("Number of major vessels (0-3) flourosopy (ca)", float(X['ca'].min()), float(X['ca'].max()), 0.0)
thal = st.sidebar.selectbox("Thalassemia (thal)", encoders['thal'].classes_)

# ================= PREDICTION =================
# Map string inputs back to Encoded values
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

# Order perfectly matches what model trained on
user_df = pd.DataFrame([user_data], columns=X.columns)
user_scaled = scaler.transform(user_df)

if st.sidebar.button("Run Diagnostics 🔍"):
    prediction = rf_model.predict(user_scaled)[0]
    
    st.markdown("### Diagnostic Results")
    if prediction == 0:
        st.success("🎉 **No Heart Disease Detected (Level 0)**")
        st.balloons()
    else:
        st.error(f"⚠️ **Heart Disease Detected - Severity Level {prediction} (out of 4)**")
        
    st.markdown("---")
    st.markdown("### Key Feature Importance driving this model")
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)
    st.bar_chart(data=importance, x='Feature', y='Importance')
    
    st.markdown("*Note: 'cp' (Chest Pain type), 'thalch' (Max heart rate) and 'exang' (Exercise induced angina) heavily dictate the model's accuracy on the UCI dataset.*")
