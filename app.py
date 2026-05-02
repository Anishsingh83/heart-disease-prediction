import streamlit as st
import pandas as pd
import joblib


model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title(" Heart Disease Prediction App")
st.markdown("### Enter patient details below")

age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

max_hr = st.slider("Max Heart Rate", 60, 200, 150)

exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")


if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

   
    input_df = input_df[expected_columns]

    
    scaled_input = scaler.transform(input_df)

    
    prediction = model.predict(scaled_input)[0]

    
    prob = model.predict_proba(scaled_input)[0][1]

  
    st.markdown("##  Prediction Result")

    if prediction == 1:
        st.error(f" High Risk of Heart Disease\n\nConfidence: {prob*100:.2f}%")
        st.progress(int(prob * 100))
    else:
        st.success(f"Low Risk of Heart Disease\n\nConfidence: {(1-prob)*100:.2f}%")
        st.progress(int((1 - prob) * 100))