import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

st.title("💼 Employee Attrition Cost Predictor")

# Load dataset

@st.cache_data
def load_data():
  df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
  df["AttritionCost"] = df["MonthlyIncome"] * 3
  return df

df = load_data()

# Features and target

X = df.drop(columns=["MonthlyIncome", "Attrition", "AttritionCost"])
y = df["AttritionCost"]

# Convert gender to numeric

X["Gender"] = X["Gender"].map({"Male":1,"Female":0})

# Detect column types

categorical_features = X.select_dtypes(include="object").columns.tolist()
numeric_features = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessing

preprocessor = ColumnTransformer(
  transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
  ]
)

pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", LinearRegression())
])

pipeline.fit(X,y)

st.subheader("Enter Employee Details")

# User Inputs

age = st.number_input("Age",18,60,30)
gender = st.selectbox("Gender",["Male","Female"])
job_role = st.selectbox("Job Role",df["JobRole"].unique())
department = st.selectbox("Department",df["Department"].unique())
business_travel = st.selectbox("Business Travel",df["BusinessTravel"].unique())

# Create input dictionary with defaults

input_dict = {}

for col in X.columns:
  if col in numeric_features:
    input_dict[col] = X[col].median()
  else:
    input_dict[col] = df[col].mode()[0]

# Override user selections

input_dict["Age"] = age
input_dict["Gender"] = 1 if gender=="Male" else 0
input_dict["JobRole"] = job_role
input_dict["Department"] = department
input_dict["BusinessTravel"] = business_travel

input_df = pd.DataFrame([input_dict])

# Prediction

if st.button("Predict Attrition Cost"):
  prediction = pipeline.predict(input_df)[0]
  st.success(f"Predicted Attrition Cost: ${prediction:,.2f}")


