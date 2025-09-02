# ML-Based-Income-Prediction-System

## Overview
A machine learning system that predicts income levels based on demographic and employment data.

## Project Structure
- `data/`: Contains the dataset files
- `src/`: Core modules for preprocessing, training, and evaluation
- `models/`: Saved ML models and encoders
- `notebooks/`: Jupyter notebooks for analysis
- `frontend/`: Streamlit web application for interactive predictions

## Setup
```
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run frontend/app.py
```

## Dataset
This project uses the Adult/Census Income dataset to predict whether income exceeds $50K/year based on census data.

## Models
The system implements and compares multiple classification algorithms including:
- Random Forest
- Gradient Boosting
- Logistic Regression