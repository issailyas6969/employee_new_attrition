from src.pipeline import create_pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(X, y, categorical_features, binary_features, numeric_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline(categorical_features=categorical_features,
                               binary_features=binary_features,
                               numeric_features=numeric_features)

    pipeline.fit(X_train, y_train)

    # Save trained pipeline
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/model.pkl")
    X_train.to_csv("model/template_columns.csv", index=False)  # Save template

    return pipeline, X_test, y_test
