import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = "model/model.pkl"
TEMPLATE_PATH = "model/template_columns.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

if not os.path.exists(TEMPLATE_PATH):
    st.error(f"Template file not found at {TEMPLATE_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
template = pd.read_csv(TEMPLATE_PATH)

st.title("Employee Attrition Cost Predictor 💼💰")

st.markdown("""
Predict the potential attrition cost of an employee based on their details.
""")

# ----------------------------
# User Input
# ----------------------------
st.sidebar.header("Employee Details")

def get_user_input():
    user_input = {}
    for col in template.columns:
        if template[col].dtype == "int64" or template[col].dtype == "float64":
            user_input[col] = st.sidebar.number_input(f"{col}", value=int(template[col].mean()))
        else:
            # If categorical, show a selectbox with unique values from template
            user_input[col] = st.sidebar.selectbox(f"{col}", template[col].unique())
    return pd.DataFrame([user_input])

input_data = get_user_input()

st.subheader("Entered Employee Data")
st.write(input_data)

# ----------------------------
# Align template
# ----------------------------
# Make sure all template columns exist after encoding
for col in template.columns:
    if col not in input_data.columns:
        input_data[col] = template[col].iloc[0]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Attrition Cost"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💸 Predicted Attrition Cost: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

