# Functions to evaluate models
# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Preprocessed Dataset
data = pd.read_csv("../data/income_cleaned.csv")

# Separate Features and Target Variable
X = data.drop("income", axis=1)
y = data["income"]

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the Trained Model
best_model = joblib.load("../models/best_income_model_pipeline.pkl")

# Generate Predictions
y_pred = best_model.predict(X_test)

# Calculate Performance Metrics
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc:.4f}")

# Generate detailed classification report 
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Create a confusion matrix to visualize prediction performance
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K"," >50K"], yticklabels=["<=50K"," >50K"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Analyze Feature Importance (If Available)

if hasattr(best_model, "feature_importances_"):
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8,5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()
else:
    print("\nℹ️ Feature importance is not available for this model.")

# Visualize Logistic Regression Coefficients
if hasattr(best_model, "coef_"):
    coefficients = pd.Series(best_model.coef_[0], index=X.columns)
    # Get top coefficients by absolute value (both positive and negative influence)
    top_coeffs = coefficients.abs().sort_values(ascending=False).head(10).index
    top_coeffs_values = coefficients[top_coeffs]
    
    plt.figure(figsize=(10,6))
    colors = ['red' if c < 0 else 'green' for c in top_coeffs_values]
    sns.barplot(x=top_coeffs_values, y=top_coeffs, palette=colors)
    plt.title("Top 10 Features by Coefficient Magnitude")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.show()

