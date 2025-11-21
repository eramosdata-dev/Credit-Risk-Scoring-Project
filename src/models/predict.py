import pandas as pd
import joblib
from src.config import MODELS_DIR
from src.utils import setup_logging

logger = setup_logging()

def predict(df: pd.DataFrame, model_path: str = None):
    """
    Make predictions using a trained model.
    
    Args:
        df (pd.DataFrame): Dataframe with features
        model_path (str): Path to the model file
        
    Returns:
        np.array: Predictions
    """
    if model_path is None:
        model_path = MODELS_DIR / "baseline_model.joblib"
        
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Preprocess (must match training)
    # Note: In a real scenario, we'd have a preprocessing pipeline saved or separate.
    # Here we assume df has the same structure as X_numeric in train.py
    # We need to select numeric columns only as we did in training
    X_numeric = df.select_dtypes(include=['number'])
    if 'TARGET' in X_numeric.columns:
        X_numeric = X_numeric.drop(columns=['TARGET'])
    if 'SK_ID_CURR' in X_numeric.columns:
        X_numeric = X_numeric.drop(columns=['SK_ID_CURR'])
        
    logger.info("Making predictions...")
    predictions = model.predict_proba(X_numeric)[:, 1]
    
    return predictions
