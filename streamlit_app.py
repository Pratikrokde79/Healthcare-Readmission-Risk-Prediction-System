import streamlit as st
import requests
import json

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# -----------------------------
# API URL (Your Deployed Endpoint)
# -----------------------------
API_URL = "https://is4mso3l84.execute-api.ap-south-1.amazonaws.com/Prod/predict"

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; color: #1A73E8;'>
        🩺 Patient Readmission Risk Predictor
    </h1>
    <p style='text-align: center; font-size: 18px; color: gray;'>
        AI-Powered Early Risk Detection • Healthcare Intelligence Dashboard
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# INPUT FORM
# -----------------------------
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=30, value=3)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=150, value=40)

with col2:
    num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=20)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=5)
    insulin = st.selectbox("Insulin Level", ["No", "Up", "Down", "Steady"])

# Convert insulin to model value
insulin_map = {"No": 0, "Up": 1, "Down": 2, "Steady": 3}
insulin_value = insulin_map[insulin]

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict Readmission Risk"):
    payload = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "insulin": insulin_value
    }

    with st.spinner("Analyzing patient risk..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            if "prediction" in result:
                risk = result["prediction"]
                probability = result.get("probability", None)

                if risk == 1:
                    st.error("⚠️ High Risk of Readmission")
                else:
                    st.success("✅ Low Risk of Readmission")

                if probability:
                    st.info(f"**Model Confidence:** {probability:.2f}")
            else:
                st.warning("Unexpected response from server.")
                st.json(result)

        except Exception as e:
            st.error("Error connecting to prediction API.")
            st.write(e)

