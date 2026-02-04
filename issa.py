from src.train_model import train_model
import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\new_employee attrition\WA_Fn-UseC_-HR-Employee-Attrition (1).csv")

# Create target
df['AttritionCost'] = df['MonthlyIncome'] * 3  # or your formula

# Features and target
X = df.drop(columns=['AttritionCost', 'MonthlyIncome', 'Attrition'])
y = df['AttritionCost']

# Encode binary columns
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Define categorical and numeric columns for the pipeline
categorical_features = ['JobRole', 'BusinessTravel', 'Department']
numeric_features = [col for col in X.columns if col not in categorical_features + ['Gender']]

# Train model (train_model will call create_pipeline)
pipeline, X_test, y_test = train_model(X, y,
                                       categorical_features=categorical_features,
                                       binary_features=['Gender'],
                                       numeric_features=numeric_features)
print("✅ Training complete!")