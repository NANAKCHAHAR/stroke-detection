import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("stroke.pkl")

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Stroke Prediction System")
st.write("Enter patient details to predict stroke risk.")

# Inputs
gender = st.selectbox("Gender", [0, 1])  # Female=0, Male=1

age = st.number_input("Age", 1, 100, 30)

hypertension = st.selectbox("Hypertension", [0, 1])

heart_disease = st.selectbox("Heart Disease", [0, 1])

ever_married = st.selectbox("Ever Married", [0, 1])

work_type = st.selectbox(
    "Work Type",
    [0, 1, 2, 3, 4]
)

residence_type = st.selectbox(
    "Residence Type",
    [0, 1]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    50.0,
    300.0,
    100.0
)

bmi = st.number_input(
    "BMI",
    10.0,
    60.0,
    25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    [0, 1, 2, 3]
)

# Prediction Button
if st.button("Predict Stroke"):

    input_data = pd.DataFrame(
        [[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status
        ]],
        columns=[
            'gender',
            'age',
            'hypertension',
            'heart_disease',
            'ever_married',
            'work_type',
            'Residence_type',
            'avg_glucose_level',
            'bmi',
            'smoking_status'
        ]
    )

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Stroke Probability: {probability*100:.2f}%")

    if probability >= 0.25:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ No Stroke Risk Detected")