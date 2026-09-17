import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):

    print("[1/4] Loading data and filtering raw columns...")
    safe_features = ['loan_amnt',
                    'term',
                    'int_rate',
                    'installment',
                    'grade',
                    'emp_length',
                    'home_ownership',
                    'annual_inc',
                    'verification_status',
                    'loan_status',
                    'purpose',
                    'dti',
                    'delinq_2yrs',
                    'fico_range_low',
                    'fico_range_high',
                    'inq_last_6mths',
                    'open_acc',
                    'pub_rec',
                    'revol_bal',
                    'revol_util',
    ]

    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)']
    paid_statuses = ['Fully Paid']
    all_target_statuses = default_statuses + paid_statuses

    df_raw = pd.read_csv(file_path, usecols=safe_features, low_memory=False)
    df_clean = df_raw[df_raw['loan_status'].isin(all_target_statuses)].copy()
    df_clean['is_default'] = df_clean['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)
    df_clean = df_clean.drop(columns=['loan_status'])

    print("[2/4] Cleaning missing values...")

    df_clean['emp_length'] = df_clean['emp_length'].fillna('Unknown')
    df_clean = df_clean.dropna(subset=['inq_last_6mths', 'dti', 'revol_util'])

    print("[3/4] Encoding text features...")

    df_clean['term'] = df_clean['term'].astype(str).str.extract('(\d+)').astype(float).astype(int)

    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df_clean['grade'] = df_clean['grade'].map(grade_mapping)

    emp_length_mapping = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
        'Unknown': -1
    }

    df_clean['emp_length'] = df_clean['emp_length'].map(emp_length_mapping)

    print("[4/4] Generating dummy columns...")
    df_final = pd.get_dummies(df_clean, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)

    print("Data cleaning and preprocessing completed.")
    return df_final

def prepare_train_test_split(df_final):
    """Separates targets and returns stratified matrices for modeling."""
    X = df_final.drop(columns=['is_default'])
    y = df_final['is_default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    import os
    sample_path = '../data/accepted_2007_to_2018Q4.csv'
    if os.path.exists(sample_path):
        cleaned_data = load_and_clean_data(sample_path)
        X_train, X_test, y_train, y_test = prepare_train_test_split(cleaned_data)
        print(f"Verified rows: Train={len(X_train):,}, Test={len(X_test):,}")