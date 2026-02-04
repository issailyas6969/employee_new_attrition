from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def create_pipeline():

    categorical_features = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"  # numeric columns
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", LinearRegression())
    ])

    return pipeline
