import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def train_and_evaluate_pipeline(model, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    try:
        # 1. Training Phase
        logger.info(f"Starting model training with {model.__class__.__name__}...")
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        
        # 2. Evaluation Phase
        logger.info("Starting model evaluation...")
        predictions = model.predict(X_test)
        
        # Replace this with your specific metric (e.g., accuracy, F1, RMSE)
        from sklearn.metrics import accuracy_score
        metric_score = accuracy_score(y_test, predictions)
        
        metrics = {"accuracy": metric_score}
        logger.info(f"Evaluation metrics calculated successfully: {metrics}")
        
        return metrics

    except ValueError as ve:
        logger.error(f"Data shape or type mismatch during training/evaluation: {str(ve)}")
        raise ve
    except Exception as e:
        logger.error(f"Unexpected failure in training/evaluation pipeline: {str(e)}", exc_info=True)
        raise e