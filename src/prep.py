import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="Input raw data")
    parser.add_argument("--clean_data", type=str, required=True, help="Output prepped data")
    return parser.parse_args()
def handle_pii(df):
    """
    HIPAA Privacy Rule: De-identify data.
    Remove direct identifiers.
    """
    # Assuming 'patient_nbr' or 'encounter_id' might be considered PII identifiers in this context if linked to real people.
    # In the UCI dataset, they are somewhat anonymized, but let's be safe and drop them for the model training.
    # If we need them for traceability, we should hash them, but here we drop them as they aren't features.
    
    pii_cols = ['encounter_id', 'patient_nbr']
    cols_to_drop = [c for c in pii_cols if c in df.columns]
    
    if cols_to_drop:
        print(f"Dropping PII/Identifier columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df
def main(args):
    print("Loading data...")
    # Handle directory input
    input_path = args.input_data
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        input_path = os.path.join(input_path, files[0])
        
    df = pd.read_csv(input_path)
    print(f"Raw shape: {df.shape}")
    # 1. PII Handling
    df = handle_pii(df)
    # 2. Handle missing medical codes / values
    # Replace '?' which is common in UCI Diabetes dataset with NaN
    df = df.replace('?', np.nan)
    
    # Drop columns with too many missing values (e.g., Weight, Payer Code often high missingness)
    # For this demo, we'll do simple imputation or drop
    missing_threshold = 0.5
    df = df.dropna(thresh=int((1-missing_threshold)*len(df)), axis=1)
    
    # Impute remaining categoricals with Mode, Numerics with Median
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    # 3. Normalize Age Groups
    # Age is often '[0-10)', '[10-20)', etc. We can map to median age or ordinal encode.
    # Let's simple Ordinal Map for simplicity
    if 'age' in df.columns:
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, 
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, 
            '[80-90)': 85, '[90-100)': 95
        }
        df['age'] = df['age'].map(age_map)
        # fallback for unmapped
        df['age'] = df['age'].fillna(65)
    # 4. Target Variable formatting
    # 'readmitted' column usually has '<30', '>30', 'NO'. 
    # Task: Predict Readmission Risk. 
    # Usually binary: Readmitted (<30 or >30) vs NO, or <30 (High Risk) vs rest.
    # Let's check unique values if column exists, else assume binary 'target'.
    target_col = 'readmitted'
    if target_col in df.columns:
        # Binary Classification: High Risk (<30 days) = 1, Low Risk (>30, NO) = 0
        df['Y'] = df[target_col].apply(lambda x: 1 if x == '<30' else 0)
        df = df.drop(columns=[target_col])
    else:
        # Fallback for demo if dataset is different
        print("Warning: 'readmitted' column not found. Creating dummy target.")
        df['Y'] = np.random.randint(0, 2, df.shape[0])
    # 5. One-Hot Encoding on Medications
    # Common med columns: 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', etc.
    # We'll just OHE all object columns remaining (except target if it was object)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"One-hot encoding: {list(cat_cols)}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"Prepped shape: {df.shape}")
    
    # Save
    output_path = args.clean_data
    if not output_path.endswith('.csv'):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, 'prepped_data.csv')
        
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
if __name__ == "__main__":
    main(parse_args())