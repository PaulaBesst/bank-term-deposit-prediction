
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and feature columns
@st.cache_resource
def load_model_and_features():
    try:
        with open('model/best_bank_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, feature_columns
    except FileNotFoundError:
        st.error("Model files not found! Please run the analysis script first.")
        return None, None

model, feature_columns = load_model_and_features()

if model is None:
    st.stop()

# App header
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.markdown("---")
st.markdown("""
**Predict whether a client will subscribe to a term deposit based on marketing campaign data.**

This model uses key demographic, economic, and campaign features to make predictions.
""")

# Sidebar for model info
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"**Features Used:** 13")
    st.info("**Model Type:** Machine Learning Classifier")
    
    st.header("📈 How to Use")
    st.markdown("""
    1. Fill in the client information
    2. Adjust campaign details
    3. Set economic indicators
    4. Click 'Predict Subscription'
    """)

# Main input form
with st.form("prediction_form"):
    st.subheader("🔍 Client & Campaign Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 DEMOGRAPHICS**")
        age = st.slider("Age", min_value=18, max_value=95, value=35, help="Client's age in years")
        
        job = st.selectbox(
            "Job Category",
            options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'],
            index=4,  # Default to management
            help="Client's job category"
        )
        
        marital = st.selectbox(
            "Marital Status",
            options=['divorced', 'married', 'single'],
            index=1,  # Default to married
            help="Client's marital status"
        )
        
        education = st.selectbox(
            "Education Level",
            options=['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
                    'professional.course', 'university.degree'],
            index=6,  # Default to university degree
            help="Client's education level"
        )
    
    with col2:
        st.markdown("**📞 CAMPAIGN DETAILS**")
        # duration = st.slider(
        #     "Call Duration (seconds)", 
        #     min_value=0, max_value=2000, value=200,
        #     help="Duration of the last contact call"
        # )
        
        campaign = st.slider(
            "Campaign Contacts", 
            min_value=1, max_value=20, value=2,
            help="Number of contacts during this campaign"
        )
        
        pdays = st.slider(
            "Days Since Last Contact", 
            min_value=-1, max_value=999, value=-1,
            help="Days since last contact (-1 means never contacted)"
        )
        
        previous = st.slider(
            "Previous Campaign Contacts", 
            min_value=0, max_value=10, value=0,
            help="Number of contacts before this campaign"
        )
        
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            options=['failure', 'nonexistent', 'success'],
            index=1,  # Default to nonexistent
            help="Outcome of the previous marketing campaign"
        )
    
    # Economic indicators section
    st.markdown("**📊 ECONOMIC INDICATORS**")
    eco_col1, eco_col2 = st.columns(2)
    
    with eco_col1:
        emp_var_rate = st.number_input(
            "Employment Variation Rate", 
            min_value=-5.0, max_value=5.0, value=1.1, step=0.1,
            help="Quarterly employment variation rate"
        )
        
        cons_price_idx = st.number_input(
            "Consumer Price Index", 
            min_value=90.0, max_value=100.0, value=93.2, step=0.1,
            help="Monthly consumer price index"
        )
    
    with eco_col2:
        cons_conf_idx = st.number_input(
            "Consumer Confidence Index", 
            min_value=-60.0, max_value=0.0, value=-36.4, step=0.1,
            help="Monthly consumer confidence index"
        )
        
        euribor3m = st.number_input(
            "Euribor 3 Month Rate", 
            min_value=0.0, max_value=10.0, value=0.7, step=0.1,
            help="Daily Euribor 3 month rate"
        )
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Subscription", use_container_width=True)

# Prediction logic
if submitted:
    try:
        # Create input dataframe with all features initialized to 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill numerical features
        if 'age' in feature_columns:
            input_data['age'] = age
        # if 'duration' in feature_columns:
        #     input_data['duration'] = duration
        if 'campaign' in feature_columns:
            input_data['campaign'] = campaign
        if 'pdays' in feature_columns:
            input_data['pdays'] = pdays
        if 'previous' in feature_columns:
            input_data['previous'] = previous
        if 'emp.var.rate' in feature_columns:
            input_data['emp.var.rate'] = emp_var_rate
        if 'cons.price.idx' in feature_columns:
            input_data['cons.price.idx'] = cons_price_idx
        if 'cons.conf.idx' in feature_columns:
            input_data['cons.conf.idx'] = cons_conf_idx
        if 'euribor3m' in feature_columns:
            input_data['euribor3m'] = euribor3m
        
        # Handle categorical features (one-hot encoded)
        categorical_mappings = {
            'job': job,
            'marital': marital,
            'education': education,
            'poutcome': poutcome
        }
        
        for category, value in categorical_mappings.items():
            # Find the corresponding one-hot encoded column
            encoded_col = f"{category}_{value}"
            if encoded_col in feature_columns:
                input_data[encoded_col] = 1
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Create result columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.success("✅ **LIKELY TO SUBSCRIBE**")
                st.balloons()
            else:
                st.error("❌ **UNLIKELY TO SUBSCRIBE**")
        
        with result_col2:
            st.metric(
                label="Subscription Probability", 
                value=f"{probability[1]:.1%}",
                help="Probability that the client will subscribe to a term deposit"
            )
        
        # Additional insights
        st.markdown("### 📋 Prediction Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("**Key Factors:**")
            factors = []
            
            # Top factors based on XGBoost importance
            if emp_var_rate > 0:
                factors.append("• Favorable employment trend (positive indicator)")
            elif emp_var_rate < -2:
                factors.append("• Economic uncertainty (may drive interest)")
            
            if previous > 0 and poutcome == 'success':
                factors.append("• Previous successful campaign (very positive)")
            elif poutcome == 'failure':
                factors.append("• Previous campaign failed (negative indicator)")
            
            if marital == 'single':
                factors.append("• Single status (higher tendency)")
            
            if factors:
                for factor in factors:
                    st.markdown(factor)
            else:
                st.markdown("• Standard profile")
        
        with insights_col2:
            st.markdown("**Recommendations:**")
            recommendations = []
            
            if probability[1] > 0.7:
                recommendations.append("• High priority contact - schedule follow-up")
                recommendations.append("• Consider premium product offerings")
            elif probability[1] > 0.4:
                recommendations.append("• Moderate potential - nurture with targeted content")
                recommendations.append("• Follow up with personalized approach")
            else:
                recommendations.append("• Low priority - include in general campaigns")
                recommendations.append("• Focus resources on higher probability clients")
            
            for rec in recommendations:
                st.markdown(rec)
        
        # Confidence indicator
        confidence_level = "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
        st.info(f"**Prediction Confidence:** {confidence_level} ({max(probability):.1%})")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please check that all required model files are present and try again.")

# Footer
st.markdown("---")
st.markdown("**Note:** This prediction model is based on historical marketing campaign data and should be used as a decision support tool alongside human judgment.")



    