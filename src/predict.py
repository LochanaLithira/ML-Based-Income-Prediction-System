# Script to take input & predict income
import pandas as pd
import joblib

pipeline_path = "models/best_income_model_pipeline.pkl"
pipeline = joblib.load(pipeline_path)

def predict_income(user_input: dict):
    """
    Predicts income class (<=50K / >50K) and probability based on user input.

    Parameters:
        user_input (dict): Keys = feature names, Values = user-provided values

    Returns:
        prediction (str): "<=50K" or ">50K"
        probability (float): Probability of income >50K
    """
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Use pipeline to predict
    pred_class = pipeline.predict(input_df)[0]
    pred_prob = pipeline.predict_proba(input_df)[0][1]  # probability of >50K
    
    return ("<=50K" if pred_class == 0 else ">50K"), pred_prob

example_input = {
    "age": 55,
    "capital-gain": 10000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "workclass": "Private",
    "education": "Undergraduate",
    "marital-status": "Married",
    "occupation": "White collar",
    "relationship": "Independent",
    "gender": "Female",
}

prediction, probability = predict_income(example_input)
print(f"Predicted Income: {prediction}")
print(f"Probability of >50K: {probability:.2f}")

