import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("student_performance_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Prediction")

st.write(
    "Predict a student's Performance Index based on "
    "academic performance and study habits."
)

st.divider()

# Input section
st.subheader("Enter Student Details")

hours_studied = st.number_input(
    "Hours Studied",
    min_value=1,
    max_value=9,
    value=5
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=40,
    max_value=99,
    value=70
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=4,
    max_value=9,
    value=7
)

sample_papers = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=9,
    value=5
)

st.divider()

# Prediction button
if st.button("Predict Performance Index", type="primary"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Hours Studied": [hours_studied],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [extracurricular],
        "Sleep Hours": [sleep_hours],
        "Sample Question Papers Practiced": [sample_papers]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.success(
        f"Predicted Performance Index: {prediction:.2f}"
    )