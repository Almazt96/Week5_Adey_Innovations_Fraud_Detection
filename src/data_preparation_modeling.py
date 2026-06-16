import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
import xgboost as xgb

# Global registry to automatically accumulate performance summaries across different runs
performance_registry = []

def save_model(model, model_name, target_dir="models"):
    """Saves a trained model artifact to the specified directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_path = os.path.join(target_dir, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    print(f"💾 Model artifact successfully saved to {file_path}")
   
def run_comprehensive_pipeline(data_path, project_name, target_column):
    print(f"\n==================================================")
    print(f"🚀 STARTING PIPELINE FOR: {project_name.upper()}")
    print(f"==================================================")
    
    # 1. Data Loading
    df = pd.read_csv(data_path)
    
    # 2. Project-Specific Feature Engineering & Target Extraction
    if project_name.lower() == "ecommerce":
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_diff_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        
        columns_to_exclude = ['signup_time', 'purchase_time', 'device_id', 'country', 'user_id']
    else:
        columns_to_exclude = []

    # Separate into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 3. Stratified Split (80/20) to preserve class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Strict Column Structural Filter (Keep only numerical metrics)
    X_train = X_train.select_dtypes(include=['number', 'bool']).drop(columns=columns_to_exclude, errors='ignore')
    X_test = X_test.select_dtypes(include=['number', 'bool']).drop(columns=columns_to_exclude, errors='ignore')

    # 5. Handle Missing Values via Training Imputer Instance
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Setup Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['f1', 'average_precision'] 
    
    # ---------------------------------------------------------
    # Baseline Model (Logistic Regression) with Cross-Validation
    # ---------------------------------------------------------
    print("\n--- 1. Training Baseline: Logistic Regression ---")
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    # Cross-validation tracking
    cv_results_lr = cross_validate(log_reg, X_train, y_train, cv=cv, scoring=scoring)
    print(f"   LogReg CV F1-Score: {np.mean(cv_results_lr['test_f1']):.4f} (+/- {np.std(cv_results_lr['test_f1']):.4f})")
    print(f"   LogReg CV AUC-PR:    {np.mean(cv_results_lr['test_average_precision']):.4f} (+/- {np.std(cv_results_lr['test_average_precision']):.4f})")
    
    # Fit final unified model and evaluate
    log_reg.fit(X_train, y_train)
    save_model(log_reg, f"logistic_baseline_{project_name.lower()}")
    
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # Ensemble Model (XGBoost) with Grid Search Tuning
    # ---------------------------------------------------------
    print("\n--- 2. Training Ensemble: Hyperparameter-Tuned XGBoost ---")
    neg_class = y_train.value_counts().iloc[0]
    pos_class = y_train.value_counts().iloc[1]
    scale_weight = neg_class / pos_class

    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_base, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='average_precision', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    print(f"   Best XGBoost Hyperparameters: {grid_search.best_params_}")
    save_model(best_xgb, f"xgboost_tuned_{project_name.lower()}")
    
    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # Holdout Evaluation & Summary Registry Mapping
    # ---------------------------------------------------------
    for model_name, y_pred, y_proba in [("Logistic Regression Baseline", y_pred_lr, y_proba_lr), 
                                        ("Tuned XGBoost Ensemble", y_pred_xgb, y_proba_xgb)]:
        
        auc_pr = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        performance_registry.append({
            "Dataset": project_name.upper(),
            "Model Architecture": model_name,
            "F1-Score": round(f1, 4),
            "AUC-PR": round(auc_pr, 4),
            "Confusion Matrix [TN, FP, FN, TP]": cm.ravel().tolist()
        })


# =========================================================
# Execution Block
# =========================================================
if __name__ == "__main__":
    # Clear out registry history before starting
    performance_registry = []

    # Run Dataset 1: E-commerce Fraud Pipeline
    run_comprehensive_pipeline("data/processed/cleaned_ecommerce.csv", "ecommerce", "class")

    # Run Dataset 2: Bank/Credit Card Pipeline
    run_comprehensive_pipeline("data/processed/cleaned_creditcard.csv", "credit_card", "Class")

    # ---------------------------------------------------------
    # Documented Comparative Outputs
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\n🏁 FINAL HIGHLIGHTED MODEL COMPARISON TABLE\n" + "="*60)
    summary_df = pd.DataFrame(performance_registry)
    print(summary_df.to_string(index=False))