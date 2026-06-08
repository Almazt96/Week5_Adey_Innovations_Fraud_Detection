# handles Data Preparation, Baseline Modeling, Ensemble Modeling (with basic tuning), 
# and Cross-Validation.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

def run_model_pipeline(data_path, target_col):
    print(f"--- Running Pipeline for {data_path} ---")
    
    # ---------------------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------------------
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # ---------------------------------------------------------
    # 2 & 4. Baseline Model (Logistic Regression) + Cross-Validation
    # ---------------------------------------------------------
    print("\nTraining Baseline: Logistic Regression...")
    # 'class_weight=balanced' is crucial for imbalanced fraud data
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    # Stratified K-Fold Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate Baseline with Cross-Validation
    scoring = ['f1', 'average_precision'] # average_precision is AUC-PR
    cv_results_lr = cross_validate(log_reg, X_train, y_train, cv=cv, scoring=scoring)
    
    print(f"LogReg CV F1-Score: {np.mean(cv_results_lr['test_f1']):.4f} (+/- {np.std(cv_results_lr['test_f1']):.4f})")
    print(f"LogReg CV AUC-PR: {np.mean(cv_results_lr['test_average_precision']):.4f} (+/- {np.std(cv_results_lr['test_average_precision']):.4f})")
    
    # Fit and test on holdout set
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # 3 & 4. Ensemble Model (XGBoost) + Basic Tuning
    # ---------------------------------------------------------
    print("\nTraining Ensemble: XGBoost...")
    # Calculate scale_pos_weight to handle imbalance in XGBoost
    neg_class = y_train.value_counts()[0]
    pos_class = y_train.value_counts()[1]
    scale_weight = neg_class / pos_class

    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
    
    # Basic Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    # Grid Search with CV
    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='average_precision', # Optimize for AUC-PR on imbalanced data
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    
    # Predict on holdout set
    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # Holdout Test Evaluation
    # ---------------------------------------------------------
    print("\n--- Final Test Set Evaluation ---")
    results = []
    
    for model_name, y_pred, y_proba in [("Logistic Regression", y_pred_lr, y_proba_lr), 
                                        ("XGBoost", y_pred_xgb, y_proba_xgb)]:
        
        auc_pr = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{model_name}:")
        print(f"AUC-PR: {auc_pr:.4f} | F1-Score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        
        results.append({
            "Model": model_name,
            "AUC-PR": auc_pr,
            "F1-Score": f1,
            "True Negatives": cm[0,0],
            "False Positives": cm[0,1],
            "False Negatives": cm[1,0],
            "True Positives": cm[1,1]
        })
        
    return pd.DataFrame(results)

# ==========================================
# Execution
# ==========================================
# Run for Credit Card Data
# results_cc = run_model_pipeline('creditcard.csv', 'Class')
# print("\nCredit Card Results Table:\n", results_cc)

# Run for Fraud Data
# results_fd = run_model_pipeline('Fraud_Data.csv', 'class')
# print("\nFraud Data Results Table:\n", results_fd)