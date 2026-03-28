
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from groq import Groq
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_readmission.pkl")
    with open("models/feature_names.json") as f:
        features = json.load(f)
    return model, features

model, feature_names = load_model()

st.title("Hospital Readmission Risk Predictor")
st.markdown("Enter patient details to predict 30-day readmission risk")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Info")
    age = st.slider("Age", 20, 95, 65)
    gender = st.selectbox("Gender", ["Female", "Male"])
    time_in_hospital = st.slider("Days in hospital", 1, 14, 4)

with col2:
    st.subheader("Clinical Details")
    num_medications = st.slider("Number of medications", 1, 40, 15)
    num_lab_procedures = st.slider("Lab procedures", 1, 100, 45)
    number_diagnoses = st.slider("Number of diagnoses", 1, 9, 5)
    num_procedures = st.slider("Procedures done", 0, 6, 1)

with col3:
    st.subheader("Visit History")
    number_inpatient = st.slider("Prior inpatient visits", 0, 15, 0)
    number_outpatient = st.slider("Prior outpatient visits", 0, 15, 0)
    number_emergency = st.slider("Prior emergency visits", 0, 15, 0)
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])

st.divider()
predict_btn = st.button("Predict Readmission Risk", type="primary",
                         use_container_width=True)

if predict_btn:
    med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    total_visits = number_inpatient + number_outpatient + number_emergency
    emergency_rate = round(number_emergency / (total_visits + 1), 3)
    medication_density = round(num_medications / (time_in_hospital + 1), 3)
    is_senior = 1 if age >= 65 else 0
    high_visit_risk = 1 if (is_senior == 1 and number_emergency >= 3) else 0

    input_data = {f: 0 for f in feature_names}
    input_data.update({
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_diagnoses': number_diagnoses,
        'num_procedures': num_procedures,
        'number_inpatient': number_inpatient,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'insulin': med_map[insulin],
        'total_visits': total_visits,
        'emergency_rate': emergency_rate,
        'medication_density': medication_density,
        'is_senior': is_senior,
        'high_visit_risk': high_visit_risk,
    })

    input_df = pd.DataFrame([input_data])
    risk_proba = model.predict_proba(input_df)[0][1]
    risk_pct = round(risk_proba * 100, 1)

    st.divider()
    r1, r2, r3 = st.columns(3)

    with r1:
        color = "red" if risk_pct > 20 else "orange" if risk_pct > 12 else "green"
        st.metric("Readmission Risk", f"{risk_pct}%")
        if risk_pct > 20:
            st.error("HIGH RISK — Immediate intervention recommended")
        elif risk_pct > 12:
            st.warning("MODERATE RISK — Close monitoring advised")
        else:
            st.success("LOW RISK — Standard discharge protocol")

    with r2:
        st.metric("Emergency Rate", f"{emergency_rate:.1%}")
        st.metric("Medication Density", f"{medication_density:.1f} meds/day")

    with r3:
        st.metric("Total Prior Visits", total_visits)
        st.metric("Senior Patient", "Yes" if is_senior else "No")

    with st.spinner("Generating SHAP explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_df)
        top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:5]
        top_factors = "\n".join([
            f"{i+1}. {feature_names[idx]}: "
            f"{'+'if shap_vals[0][idx]>0 else ''}"
            f"{shap_vals[0][idx]:.3f}"
            for i, idx in enumerate(top_idx)
        ])

    st.subheader("Top Risk Factors (SHAP)")
    fig, ax = plt.subplots(figsize=(8, 3))
    top_5_vals = shap_vals[0][top_idx]
    top_5_names = [feature_names[i] for i in top_idx]
    colors = ['#E24B4A' if v > 0 else '#378ADD' for v in top_5_vals]
    ax.barh(top_5_names[::-1], top_5_vals[::-1], color=colors[::-1])
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on risk)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    
with st.spinner("Generating AI clinical summary..."):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""You are a clinical decision support assistant.
A machine learning model assessed a diabetic patient for
30-day hospital readmission risk.

Patient: Age {age}, {time_in_hospital} days in hospital,
{num_medications} medications, {total_visits} prior visits,
emergency rate {emergency_rate:.1%}, senior: {bool(is_senior)}.

Risk: {risk_pct}% readmission probability.

Top SHAP factors:
{top_factors}

Write a concise 3-sentence clinical summary for the
attending physician. End with one clear recommendation."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ updated working model
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        summary = response.choices[0].message.content

        st.subheader("AI Clinical Summary")
        st.info(summary)

    except Exception as e:
        st.warning(f"AI summary not available: {e}")

    