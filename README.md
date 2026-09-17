# Business Problem
Developed a credit approval system for lenders to predict defaults

# Data 
Processed over 1.36 Million raw rows of LendingClub loan data. Architected an airtight data pipeline utilizing chunked extraction to eliminate information leakage, treated missing indicators safely, and optimized feature types for matrix math.

https://www.kaggle.com/datasets/wordsforthewise/lending-club/data

# Machine Learning Optimization
Trained a stratified XGBoost Classifier hitting a robust 0.7176 ROC-AUC score and an impressive 68% default capture rate (Recall), factoring in custom monetary scale-weights to maximize portfolio profitability rather than raw accuracy.

# XGBoost & SHAP
Handled complex package attribute loader updates by writing a direct model-inference gateway, isolating Loan Grade, Amortization Term, and Interest Rate as the top structural hazard indicators.

# Deployment
Packaged the serialized pipelines into a fully interactive Streamlit production dashboard complete with self-generating compliance documentation for declined applicants.