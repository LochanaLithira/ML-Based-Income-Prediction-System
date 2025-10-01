# Import Required Libraries
import pandas as pd
import numpy as np

# Load the Dataset
data = pd.read_csv("data/income.csv")
print("Initial shape:", data.shape)

selected_features = ["age","workclass","education","marital-status","occupation","relationship","gender","hours-per-week","capital-gain","capital-loss","income"]
data = data[selected_features]

# Remove Duplicate Entries
data = data.drop_duplicates()
print("Shape after removing duplicates:", data.shape)

# Handle Missing and Categorical Data
# Replace placeholder values (?) with proper NaN values 
data = data.replace("?", np.nan)

# Define categorical columns for processing
categorical_cols = ["workclass", "education", "marital-status", 
                    "occupation", "relationship", 
                    "gender"]

# Fill missing categorical values with "Unknown" instead of dropping rows
for col in categorical_cols:
    data[col] = data[col].fillna("Unknown")

# Convert income target variable to binary format (1 for >50K, 0 for <=50K)
data["income"] = data["income"].astype(str).str.strip()
data["income"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)

# Verify no missing values remain in the dataset
print("Missing values after handling:\n", data.isnull().sum())

# Handle Outliers in Numeric Features
numeric_cols_iqr = ["age", "hours-per-week"]   # Apply IQR here
skewed_cols = ["capital-gain", "capital-loss"] # Transform instead of remove

# Function: remove outliers with IQR
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# Apply IQR only on selected numeric cols
for col in numeric_cols_iqr:
    before = data.shape[0]
    data = remove_outliers_iqr(data, col)
    after = data.shape[0]
    print(f"{col}: removed {before - after} outliers")

# Apply log transformation to skewed features
import numpy as np
for col in skewed_cols:
    data[col] = np.log1p(data[col])   # log(1 + x) keeps 0 as 0
    print(f"{col}: applied log transformation")

import pandas as pd

def preprocess_income_data(df):
    # --------------------
    # Workclass
    # --------------------
    workclass_map = {
        'Private': 'Private',
        'Self-emp-not-inc': 'Private',
        'Self-emp-inc': 'Private',
        'Federal-gov': 'Government',
        'State-gov': 'Government',
        'Local-gov': 'Government',
        'Without-pay': 'Other/Unemployed',
        'Never-worked': 'Other/Unemployed',
        '?': 'Other/Unemployed',
        'Unknown': 'Unknown'
    }
    df['workclass'] = df['workclass'].map(workclass_map)

    # --------------------
    # Education
    # --------------------
    education_map = {
    'Preschool': 'School',
    '1st-4th': 'School',
    '5th-6th': 'School',
    '7th-8th': 'School',
    '9th': 'School',
    '10th': 'School',
    '11th': 'School',
    '12th': 'School',

    'HS-grad': 'High School',

    'Some-college': 'Undergraduate',
    'Assoc-acdm': 'Undergraduate',
    'Assoc-voc': 'Undergraduate',
    'Bachelors': 'Undergraduate',

    'Masters': 'Postgraduate',
    'Doctorate': 'Postgraduate',
    'Prof-school': 'Postgraduate'
}

    df['education'] = df['education'].map(education_map)

    # --------------------
    # Marital Status
    # --------------------
    marital_map = {
        'Never-married': 'Single',
        'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married',
        'Married-spouse-absent': 'Married',
        'Divorced': 'Previously married',
        'Separated': 'Previously married',
        'Widowed': 'Previously married'
    }
    df['marital-status'] = df['marital-status'].map(marital_map)

    # --------------------
    # Occupation
    # --------------------
    occupation_map = {
        'Craft-repair': 'Blue collar',
        'Transport-moving': 'Blue collar',
        'Handlers-cleaners': 'Blue collar',
        'Farming-fishing': 'Blue collar',
        'Machine-op-inspct': 'Blue collar',
        'Other-service': 'Service',
        'Priv-house-serv': 'Service',
        'Protective-serv': 'Service',
        'Exec-managerial': 'White collar',
        'Prof-specialty': 'White collar',
        'Sales': 'White collar',
        'Adm-clerical': 'White collar',
        'Tech-support': 'White collar',
        'Armed-Forces': 'Military',
        '?': 'Unknown',
        'Unknown': 'Unknown'
    }
    df['occupation'] = df['occupation'].map(occupation_map)

    # --------------------
    # Relationship
    # --------------------
    relationship_map = {
        'Husband': 'Partnered',
        'Wife': 'Partnered',
        'Own-child': 'Child',
        'Other-relative': 'Other family',
        'Not-in-family': 'Independent',
        'Unmarried': 'Independent'
    }
    df['relationship'] = df['relationship'].map(relationship_map)

    return df

# Keep only selected features
data = data[selected_features]

# Preprocess categorical features
data = preprocess_income_data(data)

# Check cleaned categories
print(data['workclass'].value_counts())
print(data['education'].value_counts())
print(data['occupation'].value_counts())

# Verify Final Dataset
print("\nFinal dataset shape:", data.shape)
print(data.head())

# Save Preprocessed Dataset
import os
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "income_cleaned.csv")
data.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved as '{output_path}'")