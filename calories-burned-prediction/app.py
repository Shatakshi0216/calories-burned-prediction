import streamlit as st
import numpy as np

# Paste your trained weights and bias here (after training)
weights = [0.0023, 0.1445, 0.5602, 0.0481]  # Replace with your final weights
bias = 1.1523  # Replace with your final bias

def predict_new_input(new_data, weights, bias):
    y_pred = sum([weights[i] * new_data[i] for i in range(len(weights))]) + bias
    return y_pred

st.title("🔥 Calories Burned Prediction App")
st.markdown("This app predicts the number of calories burned during a workout based on your input.")

# User inputs
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
duration = st.number_input("Duration of Workout (in minutes)", min_value=1.0, max_value=300.0, value=30.0)
heart_rate = st.number_input("Heart Rate", min_value=60, max_value=200, value=130)

# Convert gender to numeric
gender_numeric = 0 if gender == "Female" else 1

# Predict button
if st.button("Predict Calories Burned"):
    input_data = [gender_numeric, age, duration, heart_rate]
    result = predict_new_input(input_data, weights, bias)
    st.success(f"🔥 Estimated Calories Burned: **{result:.2f}**")

