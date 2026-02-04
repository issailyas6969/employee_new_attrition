import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

st.title("💼 Employee Attrition Cost Predictor")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df['AttritionCost'] = df['MonthlyIncome'] * 3  # Example formula
    return df

df = load_data()

# Features and target
X = df.drop(columns=['MonthlyIncome', 'Attrition', 'AttritionCost'])
y = df['AttritionCost']

# Convert Gender to numeric
X['Gender'] = X['Gender'].map({'Male':1, 'Female':0})

# Define categorical features
categorical_features = ['JobRole','BusinessTravel','Department']
binary_features = ['Gender']

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num','passthrough', [col for col in X.columns if col not in categorical_features + binary_features])
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train pipeline
pipeline.fit(X, y)

st.subheader("Enter Employee Details:")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=60, value=30)
gender = st.selectbox("Gender", ['Male','Female'])
job_role = st.selectbox("Job Role", X['JobRole'].unique())
department = st.selectbox("Department", X['Department'].unique())
business_travel = st.selectbox("Business Travel", X['BusinessTravel'].unique())

# Prepare input for prediction
input_df = pd.DataFrame({
    'Age':[age],
    'Gender':[1 if gender=='Male' else 0],
    'JobRole':[job_role],
    'Department':[department],
    'BusinessTravel':[business_travel]
})

# Fill remaining numeric columns with median
for col in [c for c in X.columns if c not in input_df.columns]:
    input_df[col] = X[col].median()

# Predict
if st.button("Predict Attrition Cost"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted Attrition Cost: ${prediction:,.2f}")

