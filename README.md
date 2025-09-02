# 💰 ML-Based-Income-Prediction-System

## ❓ Problem
Financial institutions need to know whether a person earns more than 50,000 USD per year to make decisions about loans, credit, and risk assessment. Manually checking this for every individual is slow and error-prone.

## 💡 Solution
We built a Machine Learning system that predicts a person’s income category (<=50K or >50K) using demographic and work-related data (age, education, occupation, hours per week, etc.). The system cleans and preprocesses the data, trains models like **Decision Tree**, **Random Forest**, and **Logistic Regression**, and evaluates their accuracy.

## 🎯 Outcome
The system can automatically predict income levels, helping organizations make fast and accurate financial decisions. Users can also try the prediction through a simple interactive frontend built with **Streamlit**.

<br>

## 🛠 Technologies Used – Income Prediction System

- **💻 Programming Language:** Python
- **📊 Data Handling:** pandas, numpy
- **📈 Data Visualization:** matplotlib, seaborn
- **🤖 Machine Learning & Modeling:** scikit-learn
  - Logistic Regression
  - Decision Tree
  - Random Forest
- **💾 Model Saving:** joblib or pickle
- **🖥 Frontend / Interactive Demo:** Streamlit

<br>

# 💻 Software Implementation – Main Tasks
1️⃣
## 1️⃣ Data Preprocessing
- Clean and transform raw data
- Handle missing values, duplicates, and incorrect data types
- Encode categorical variables
- Normalize/scale numerical features

## 2️⃣ Exploratory Data Analysis (EDA) 🔍
- **Visualize feature distributions**: histograms, boxplots  
- **Analyze relationships** between features and income  
- **Identify patterns** to improve model performance  

## 3️⃣ Model Training 🏋️‍♂️
- Train machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)  
- Tune hyperparameters for optimal performance  
- Save the trained model for later use  

## 4️⃣ Backend Prediction Script 📝
Create a script `predict.py` that:  
- Loads the saved model  
- Accepts user inputs (age, education, occupation, hours per week, etc.)  
- Outputs predicted income category (`<=50K` or `>50K`)  

## 5️⃣ Frontend (Interactive Demo) 🌐
Build a **Streamlit app** `app.py`:  
- Input fields for user attributes  
- Button to predict income  
- Display predicted income category in real-time  

## 7️⃣ Testing ✅
- Test the full workflow:  
  `Data preprocessing → model prediction → frontend output`  
- Ensure predictions are accurate and the system runs smoothly
