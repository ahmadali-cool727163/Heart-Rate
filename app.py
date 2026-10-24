import pandas as pd
import streamlit as st
from model_utils import train, FEATURES

st.set_page_config(page_title="Heart Risk Checker", page_icon="❤️")


@st.cache_resource
def get_model():
    return train()  # trains in <1 second, so no .pkl version problems


model, test_acc = get_model()

st.title("❤️ Heart Risk Checker")
st.caption("A student ML project using the UCI Cleveland heart disease dataset "
           "(Logistic Regression, 13 features).")

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda x: {
            1: "Typical angina", 2: "Atypical angina",
            3: "Non-anginal pain", 4: "Asymptomatic"}[x])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {
            0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[x])
    with c2:
        thalach = st.number_input("Maximum heart rate achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-induced angina", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST", [1, 2, 3], format_func=lambda x: {
            1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
        ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {
            3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]], columns=FEATURES)
    prob = model.predict_proba(row)[0][1]
    st.metric("Estimated risk", f"{prob:.0%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.error("Heart disease likely. Please consult a doctor.")
    else:
        st.success("Heart disease unlikely.")

st.divider()
st.caption(f"Model accuracy on held-out test data: {test_acc:.0%}. "
           "Educational demo only, not medical advice.")
