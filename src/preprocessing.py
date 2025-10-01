# Import Required Libraries
import pandas as pd
import numpy as np

# Load the Dataset
data = pd.read_csv("../data/income.csv")
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
# Define numeric columns that need outlier treatment
numeric_cols = ["age","capital-gain", "capital-loss", "hours-per-week"]

# Define function to remove outliers using IQR method
def remove_outliers_iqr(df, col):
    # Calculate first and third quartiles
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    # Calculate interquartile range
    IQR = Q3 - Q1
    # Define bounds for outlier detection
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Return only rows within the acceptable range
    return df[(df[col] >= lower) & (df[col] <= upper)]

# Apply outlier removal to each numeric column
for col in numeric_cols:
    before = data.shape[0]
    data = remove_outliers_iqr(data, col)
    after = data.shape[0]
    print(f"{col}: removed {before - after} outliers")

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
data.to_csv("../data/income_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as '../data/income_cleaned.csv'")