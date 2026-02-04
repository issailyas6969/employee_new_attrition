
import pandas as pd

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    drop_cols = [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "Attrition"   # ❗ VERY IMPORTANT
    ]
    df = df.drop(columns=drop_cols)


    df['AttritionCost'] = df['MonthlyIncome'] * 3  # replace 3 with months to replace
# optionally add TrainingCost + ProductivityLoss


    X = df.drop(columns=['AttritionCost', 'MonthlyIncome'])  # features

    y = df['AttritionCost']


    return X, y

