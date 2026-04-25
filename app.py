import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("features.pkl")

# ==========================================
# HEADER
# ==========================================
st.title("🧠 Hon. Selasi's Stroke Detector")
st.markdown("### Powered by Living_EHF Model")
st.write("Enter patient clinical details to assess stroke risk.")
st.markdown("### NOTE: 1 = Yes while 0 = No")

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================
st.subheader("📋 Patient Information")

age = st.slider("Age", 1, 100, 50)

col1, col2 = st.columns(2)

with col1:
    bp = st.selectbox("High Blood Pressure", [0,1], help="1 = Yes, 0 = No")
    chest_pain = st.selectbox("Chest Pain", [0,1])
    breath = st.selectbox("Shortness of Breath", [0,1])
    snore = st.selectbox("Sleeping", [0,1])

with col2:
    chest_discomfort = st.selectbox("Chest Discomfort", [0,1])
    heartbeat = st.selectbox("Irregular Heartbeat", [0,1])
    fatigue = st.selectbox("Fatigue / Weakness", [0,1])
    dizziness = st.selectbox("Dizziness", [0,1])

cold = st.selectbox("Cold Hands/Feet", [0,1])

st.divider()

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict Stroke Risk"):

    input_data = pd.DataFrame([{
        'age': age,
        'high_blood_pressure': bp,
        'chest_pain': chest_pain,
        'shortness_of_breath': breath,
        'snoring_sleep_apnea': snore,
        'chest_discomfort': chest_discomfort,
        'irregular_heartbeat': heartbeat,
        'fatigue_weakness': fatigue,
        'dizziness': dizziness,
        'cold_hands_feet': cold
    }])

    for col in selected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[selected_features]
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[:,1][0]

    # ==========================================
    # RISK LEVEL CLASSIFICATION
    # ==========================================
    if prob < 0.30:
        risk_level = "LOW"
        color = "green"
        message = "Low risk. Maintain a healthy lifestyle."
    elif prob < 0.60:
        risk_level = "MEDIUM"
        color = "orange"
        message = "Moderate risk. Consider medical check-up."
    else:
        risk_level = "HIGH"
        color = "red"
        message = "High risk. Seek medical attention."

    st.divider()

    st.markdown("## 🧾 Prediction Result")

    st.markdown(f"### 🎯 Risk Level: **:{color}[{risk_level}]**")
    st.markdown(f"### 📊 Probability: **{prob:.4f}**")

    if risk_level == "HIGH":
        st.error(message)
    elif risk_level == "MEDIUM":
        st.warning(message)
    else:
        st.success(message)

    st.info("⚠️ Dear user, kindly beware this system is for decision support only and not solely a medical diagnosis.")