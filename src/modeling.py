import os
import joblib

def save_model(model, model_name, target_dir="models"):
    """
    Saves a trained model artifact to the specified directory.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
        
    file_path = os.path.join(target_dir, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    print(f"Model successfully saved to {file_path}")

# Example Usage after training your tuned XGBoost model:
# save_model(tuned_xgb_model, "xgb_fraud_detection_v1")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_modeling_pipeline(data_path, dataset_name, target_column):
    """
    Executes the entire training pipeline for a specific dataset.
    """
    print(f"\n" + "="*40)
    print(f"Starting Modeling Pipeline for: {dataset_name.upper()}")
    print("="*40)
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 2. Stratified Split (Addresses class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Baseline Model (Logistic Regression)
    print(f"\n--- Training Baseline Logistic Regression ({dataset_name}) ---")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    print(classification_report(y_test, lr_preds))
    save_model(lr_model, f"logistic_baseline_{dataset_name}")
    
    # 4. Advanced Model (XGBoost)
    print(f"\n--- Training Tuned XGBoost ({dataset_name}) ---")
    # (Insert your tuned hyperparameters here)
    xgb_model = XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    print(classification_report(y_test, xgb_preds))
    save_model(xgb_model, f"xgboost_tuned_{dataset_name}")

# --- Execution Example ---
# run_modeling_pipeline("data/processed/ecommerce_fraud.csv", "ecommerce", "class")
# run_modeling_pipeline("data/processed/credit_card_fraud.csv", "credit_card", "Class")