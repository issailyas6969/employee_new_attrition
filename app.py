import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("model/model.pkl")

# Load template columns
template = pd.read_csv("model/template_columns.csv")

st.title("Employee Attrition Cost Predictor 💼💰")

# --- User input ---
user_input = {
    "Age": st.number_input("Age", min_value=18, max_value=60, value=30),
    "BusinessTravel": st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]),
    "Department": st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"]),
    "DistanceFromHome": st.number_input("Distance From Home (km)", min_value=1, max_value=50, value=5),
    "Education": st.selectbox("Education Level", [1,2,3,4,5]),
    "EducationField": st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "JobRole": st.selectbox(
        "Job Role",
        ["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"]
    ),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "OverTime": st.selectbox("OverTime", ["Yes", "No"]),
    "TotalWorkingYears": st.number_input("Total Working Years", min_value=0, max_value=40, value=5),
    "YearsAtCompany": st.number_input("Years at Company", min_value=0, max_value=40, value=3)
}

# --- Fill template with user input ---
input_data = template.copy()
for key, value in user_input.items():
    if key in input_data.columns:
        input_data[key] = value

# --- Predict ---
if st.button("Predict Attrition Cost"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Attrition Cost if this employee leaves: ₹{prediction:.2f}")
