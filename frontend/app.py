# Streamlit frontend for interactive demo
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add parent directory to path so we can import from the src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# App configuration
st.set_page_config(
    page_title="Income Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Form section styling */
    .form-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08), 0 8px 16px rgba(0,0,0,0.04);
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
    }
    
    .result-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .result-income {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-income.high {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .result-income.low {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .confidence-section {
        background: rgba(102, 126, 234, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .confidence-percentage {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .confidence-label {
        font-size: 1.1rem;
        color: #4a5568;
        font-weight: 500;
    }
    
    .prediction-details {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .detail-item {
        background: rgba(247, 250, 252, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .detail-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .detail-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    .progress-container {
        margin: 1.5rem 0;
        background: #f7fafc;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .progress-label {
        font-size: 0.9rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field styling */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>💰 Income Prediction System</h1>
        <p>Advanced machine learning model to predict whether individuals earn more than $50K annually</p>
    </div>
    """, unsafe_allow_html=True)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    """Load the trained model pipeline and cache it for efficiency"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "best_income_model_pipeline.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"⚠️ Model file not found at: {model_path}")
            st.info("Please train the model first by running the model_training.ipynb notebook.")
            available_files = os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else []
            st.write(f"Files in models directory: {available_files}")
            return None
        
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Try running the model_training.ipynb notebook first to create the model file.")
        return None

# Get the model
pipeline = load_model()

if pipeline is None:
    st.stop()

# Function to make predictions
def predict_income(user_input):
    """Predicts income class and probability based on user input"""
    input_df = pd.DataFrame([user_input])
    pred_class = pipeline.predict(input_df)[0]
    pred_prob = pipeline.predict_proba(input_df)[0][1]
    earns_more_than_50k = pred_class == 1
    return earns_more_than_50k, pred_prob

# Create form for user inputs
with st.form("prediction_form"):
    # Personal Information Section
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=79, value=35)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
    
    with col2:
        marital_status = st.selectbox(
            "Marital Status",
            options=[
                "Married", "Previously married", "Single"
            ],
            index=0
        )
        relationship = st.selectbox(
            "Relationship",
            options=[
                "Child", "Independent", "Other family", "Partnered"
            ],
            index=0
        )
    
    with col3:
        education = st.selectbox(
            "Education Level",
            options=[
                "High School", "Postgraduate", "School", "Undergraduate"
            ],
            index=0
        )
    
    st.markdown("---")
    
    # Employment Information Section
    st.markdown('<div class="section-header">💼 Employment Information</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    
    with col4:
        workclass = st.selectbox(
            "Work Class",
            options=[
                "Government", "Other/Unemployed", "Private", "Unknown"
            ],
            index=2  # Default to "Private" (index 2)
        )
    
    with col5:
        occupation = st.selectbox(
            "Occupation",
            options=[
                "Blue collar", "Military", "Service", "Unknown", "White collar"
            ],
            index=0
        )
    
    with col6:
        hours_per_week = st.number_input(
            "Hours per Week", 
            min_value=25,
            max_value=57,
            value=40
        )
    
    st.markdown("---")
    
    # Financial Information Section
    st.markdown('<div class="section-header">💵 Financial Information</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    
    with col7:
        capital_gain = st.number_input(
            "Capital Gain ($)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000,
            help="Income from investment sources"
        )
    
    with col8:
        capital_loss = st.number_input(
            "Capital Loss ($)",
            min_value=0,
            max_value=10000,
            value=0,
            step=100,
            help="Losses from investment sources"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit button
    submit_button = st.form_submit_button(label="🔮 Predict Income Level")

# Make prediction when form is submitted
if submit_button:
    user_input = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "gender": gender,
        "hours-per-week": hours_per_week,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss
    }
    
    with st.spinner("🔄 Analyzing data..."):
        earns_more_than_50k, probability = predict_income(user_input)
    
    # Display professional results
    st.markdown("<br>", unsafe_allow_html=True)
    
   
    
    # Header
    st.markdown('<div class="result-header">🎯 Income Prediction Analysis</div>', unsafe_allow_html=True)
    
    # Main prediction result
    if earns_more_than_50k:
        st.markdown(f'''
        <div class="result-income high">
            ✅ Earns More Than 50,000 LKR
        </div>
        ''', unsafe_allow_html=True)
        prediction_class = "High Income Earner"
        prediction_icon = "💰"
        result_color = "#48bb78"
        income_status = "YES - Predicted to earn more than $50,000 annually"
    else:
        st.markdown(f'''
        <div class="result-income low">
            ❌ Does Not Earn More Than 50,000 LKR
        </div>
        ''', unsafe_allow_html=True)
        prediction_class = "Standard Income Earner"
        prediction_icon = "💵"
        result_color = "#d40a03"
        income_status = "NO - Predicted to earn $50,000 or less annually"
    
    # Confidence section
    st.markdown(f'''
    <div class="confidence-section">
        <div class="confidence-percentage">{probability*100:.1f}%</div>
        <div class="confidence-label">Model Confidence Level</div>
    </div>
    ''', unsafe_allow_html=True)
    
# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 0.9rem;'>© 2025 Income Prediction System | Powered by Machine Learning</p>",
    unsafe_allow_html=True
)