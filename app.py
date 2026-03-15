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
  df['AttritionCost'] = df['MonthlyIncome'] * 3
  return df

df = load_data()

# Features and target

X = df.drop(columns=['MonthlyIncome', 'Attrition', 'AttritionCost'])
y = df['AttritionCost']

# Convert Gender to numeric

X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Automatically detect categorical columns

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Column Transformer

preprocessor = ColumnTransformer(
transformers=[
  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
  ('num', 'passthrough', numeric_features)
]
)

# Pipeline

pipeline = Pipeline([
  ('preprocessor', preprocessor),
  ('regressor', LinearRegression())
])

# Train model

pipeline.fit(X, y)

st.subheader("Enter Employee Details:")

# User Inputs

age = st.number_input("Age", 18, 60, 30)
gender = st.selectbox("Gender", ['Male', 'Female'])
job_role = st.selectbox("Job Role", df['JobRole'].unique())
department = st.selectbox("Department", df['Department'].unique())
business_travel = st.selectbox("Business Travel", df['BusinessTravel'].unique())

# Prepare input dataframe

input_data = {col: X[col].median() for col in numeric_features}

input_data.update({
  'Age': age,
  'Gender': 1 if gender == 'Male' else 0,
  'JobRole': job_role,
  'Department': department,
  'BusinessTravel': business_travel
})

input_df = pd.DataFrame([input_data])

# Prediction

if st.button("Predict Attrition Cost"):
  prediction = pipeline.predict(input_df)[0]
  st.success(f"Predicted Attrition Cost: ${prediction:,.2f}")

