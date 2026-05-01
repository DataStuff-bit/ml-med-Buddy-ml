import streamlit as st
import requests
from dotenv import load_dotenv
import os
load_dotenv()
API_URL = os.getenv("API_URL")

st.set_page_config(page_title="MedBuddy Heart Disease Predictor")

st.title("❤️ MedBuddy - Heart Disease Prediction")

st.markdown("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120.0)
chol = st.number_input("Cholesterol", value=190.0)
fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=170.0)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", value=0.5)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# Button
if st.button("Predict"):
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Successful!")

            st.write("### Result")
            st.write(f"Prediction: {result['prediction']}")
            st.write(f"Probability: {result['probability'] * 100:.0f}%")
            st.write(f"Diagnosis: {result['diagnosis']}")

        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")