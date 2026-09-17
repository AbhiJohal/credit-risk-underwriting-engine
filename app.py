import streamlit as strl
import pandas as pd
import numpy as np
import joblib

# Set up clean web page configurations
strl.set_page_config(page_title="FinTech Credit Risk Engine", layout="wide")

# 1. Load the trained model assets safely
@strl.cache_resource # Keeps the model in memory so it doesn't reload on every click
def load_model_assets():
    model = joblib.load('models/credit_xgb_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return model, feature_names

try:
    model, feature_names = load_model_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False

# 2. Main Dashboard Interface Setup
strl.title("🏦 Institutional Credit Risk & Compliance Engine")
strl.markdown("Input application parameters to generate automated underwriting decisions and compliance documentation.")

if not assets_loaded:
    strl.error("Could not find model files. Please run 'python src/train.py' first to generate your model assets.")
else:
    # Organize layout into two visual side-by-side columns
    col1, col2 = strl.columns([1, 1.5])
    
    with col1:
        strl.header("Applicant Credit Profile")
        
        # Numeric input sliders and boxes
        loan_amnt = strl.number_input("Requested Loan Amount ($)", min_value=1000, max_value=40000, value=10000, step=500)
        term = strl.selectbox("Loan Term Period", options=[36, 60], format_func=lambda x: f"{x} Months")
        int_rate = strl.slider("Assigned Interest Rate (%)", min_value=4.0, max_value=32.0, value=12.5, step=0.1)
        installment = strl.number_input("Calculated Monthly Installment ($)", min_value=10.0, max_value=1500.0, value=330.0)
        grade = strl.slider("Underwriting Grade (A=Safe, G=Risky)", min_value=1, max_value=7, value=3)
        
        annual_inc = strl.number_input("Verified Borrower Annual Income ($)", min_value=5000, max_value=500000, value=65000, step=1000)
        dti = strl.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=100.0, value=18.5, step=0.1)
        fico_high = strl.slider("Borrower FICO Score (Upper Range)", min_value=500, max_value=850, value=710)
        fico_low = fico_high - 4 # Standardize lower range
        emp_length = strl.slider("Employment Stability Length (Years)", min_value=-1, max_value=10, value=5)
        # Categorical choices matching our LendingClub columns
        home_ownership = strl.selectbox("Home Ownership Status", options=['MORTGAGE', 'RENT', 'OWN', 'ANY'])
        verification_status = strl.selectbox("Income Verification Status", options=['Source Verified', 'Verified', 'Not Verified'])
        purpose = strl.selectbox("Primary Purpose of Loan", options=['debt_consolidation', 'credit_card', 'home_improvement', 'other'])

        # Unused structural placeholders initialized to zero to keep dimensions correct
        delinq_2yrs = 0.0
        inq_last_6mths = 0.0
        open_acc = 10.0
        pub_rec = 0.0
        revol_bal = 15000.0
        revol_util = 45.0

    with col2:
        strl.header("Underwriting Decision Gateway")
        
        # 3. Process inputs into the exact column matrix shape expected by XGBoost
        input_data = {
            'loan_amnt': [loan_amnt], 'term': [term], 'int_rate': [int_rate], 'installment': [installment],
            'grade': [grade], 'emp_length': [emp_length], 'annual_inc': [annual_inc], 'dti': [dti],
            'delinq_2yrs': [delinq_2yrs], 'fico_range_low': [fico_low], 'fico_range_high': [fico_high],
            'inq_last_6mths': [inq_last_6mths], 'open_acc': [open_acc], 'pub_rec': [pub_rec],
            'revol_bal': [revol_bal], 'revol_util': [revol_util]
        }
        
        # Set up one-hot encoded dummy column rules mimicking pd.get_dummies
        for feature in feature_names:
            if feature not in input_data:
                # Dynamically set categories to 1 if matched, else 0
                if 'home_ownership_' in feature and feature.split('_')[-1] == home_ownership:
                    input_data[feature] = [1]
                elif 'verification_status_' in feature and feature.split('_')[-1] == verification_status:
                    input_data[feature] = [1]
                elif 'purpose_' in feature and feature.split('_', 1)[-1] == purpose:
                    input_data[feature] = [1]
                else:
                    input_data[feature] = [0]
                    
        input_df = pd.DataFrame(input_data)[feature_names]
        
        # 4. Generate Calculations
        prob_default = model.predict_proba(input_df)[0, 1]
        
        # Display large status metric cards
        strl.metric(label="Calculated Default Probability Score", value=f"{prob_default:.2%}")
        
        # Establish conservative credit tier thresholds
        if prob_default < 0.30:
            strl.success("APPLICATION APPROVED: Low Credit Risk Profile")
        elif prob_default < 0.60:
            strl.warning("REFER TO CREDIT COMMITTEE: Borderline Risk Tier")
        else:
            strl.error("APPLICATION DENIED: High Structural Credit Risk")
            
            # 5. Output Regulatory Compliance Adverse Action Notice
            strl.markdown("---")
            strl.subheader("🔒 Automated Regulatory Compliance Documentation")
            strl.caption("Generated automatically pursuant to the Equal Credit Opportunity Act (ECOA)")
            
            notice_text = f"""
            **🏦 ADVERSE ACTION NOTICE OF CREDIT DENIAL**  
            **Date:** Automated System Clock  
            
            Dear Applicant,  
            Thank you for your recent application. After careful electronic evaluation of your credit capacity, 
            we regret to inform you that we are unable to extend credit approval terms at this time. 
            
            Our statistical underwriting model isolated the following features as adding significant default risk vectors to your profile:
            *   **LOAN TERM MULTIPLIER:** Requesting longer amortisation horizons ({term} months) significantly spikes long-term default hazard ratios.
            *   **DEBT-TO-INCOME AND PRICING MATRIX:** The current combination of your DTI ({dti}%) paired with the structural sizing of your requested loan amount (${loan_amnt:,}) exceeds safe operational capital boundaries.
            """
            strl.info(notice_text)
