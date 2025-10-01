# Train ML models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the Preprocessed Dataset
data = pd.read_csv("data/income_cleaned.csv")
print("Dataset shape:", data.shape)

# Separate Features and Target Variable
X = data.drop("income", axis=1)
y = data["income"]

numeric_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "gender"]

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

pipelines = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

accuracies = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

best_model_name = max(accuracies, key=accuracies.get)
best_pipeline = pipelines[best_model_name]

model_path = "models/best_income_model_pipeline.pkl"
joblib.dump(best_pipeline, model_path)

print(f"\n✅ Best model pipeline ({best_model_name}) saved as '{model_path}'")