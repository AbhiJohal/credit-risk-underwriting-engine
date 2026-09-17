import os
import xgboost as xgb
# We use joblib to save our trained model file to the disk
import joblib

# Import the code we wrote in our previous script file
from data_pipeline import load_and_clean_data, prepare_train_test_split

def train_credit_engine(X_train, y_train):
    """
    Initializes and fits an XGBoost classifier optimized for imbalanced financial risk.
    """
    print("Training the production XGBoost Credit Risk Engine...")
    
    # Using the same proven parameters from our notebook phase
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=3.7, 
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def save_artifacts(model, feature_names):
    """Saves the trained model and feature list to the disk."""
    # Ensure a directory named 'models' exists to store our output
    os.makedirs('models', exist_ok=True)
    
    # Save the model file
    model_path = 'models/credit_xgb_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")
    
    # Save feature names so our dashboard knows the column order later
    feature_path = 'models/feature_names.joblib'
    joblib.dump(feature_names, feature_path)
    print(f"Feature column map saved to: {feature_path}")

if __name__ == "__main__":
    # Define your exact raw file path
    FILE_PATH = 'data/accepted_2007_to_2018Q4.csv'
    
    if os.path.exists(FILE_PATH):
        # 1. Run pipeline to clean data
        df_final = load_and_clean_data(FILE_PATH)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = prepare_train_test_split(df_final)
        
        # 3. Train model
        model = train_credit_engine(X_train, y_train)
        
        # 4. Save the finished model file
        save_artifacts(model, X_train.columns.tolist())
    else:
        print(f"Error: Could not find raw data file at {FILE_PATH}. Please check your filename configuration.")
