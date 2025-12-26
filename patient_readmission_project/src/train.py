import argparse
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True, help="Path to prepped data")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save the model")
    parser.add_argument("--target_column", type=str, default="Y", help="Target column name")
    return parser.parse_args()

def main(args):
    # Enable automatic logging for scikit-learn
    mlflow.autolog()
    
    print("--- Start Training Process ---")
    
    # 1. Handle Input Path (Azure ML can pass a folder or a file)
    input_path = args.training_data
    if os.path.isdir(input_path):
        print(f"Input is a directory: {input_path}")
        files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No CSV files found in {input_path}")
        input_path = os.path.join(input_path, files[0])
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. Identify Target Column
    target_col = args.target_column
    if target_col not in df.columns:
        print(f"Warning: Target '{target_col}' not found. Using the last column instead.")
        target_col = df.columns[-1]
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train Model
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    print(f"Metrics: Accuracy={acc:.4f}, AUC={roc_auc:.4f}, F1={f1:.4f}, PR_AUC={pr_auc:.4f}")
    
    # 6. Explicitly Log Metrics for Azure ML UI
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("AUC", roc_auc)
    mlflow.log_metric("F1", f1)
    
    # 7. Generate and Save PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("pr_curve.png")
    mlflow.log_artifact("pr_curve.png")
    
    # 8. Save Model to the output path required by the pipeline
    # Azure ML Mount points require the folder to exist first
    if not os.path.exists(args.model_output):
        os.makedirs(args.model_output, exist_ok=True)
    
    print(f"Saving model to: {args.model_output}")
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)
    
    print("--- Training Complete ---")

if __name__ == "__main__":
    main(parse_args())
