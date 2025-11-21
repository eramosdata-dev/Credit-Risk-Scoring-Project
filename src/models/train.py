import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from src.utils import setup_logging
from src.config import MODELS_DIR
import joblib

logger = setup_logging()

def train_model(df: pd.DataFrame, target_col: str = 'TARGET'):
    """
    Train a baseline model.
    
    Args:
        df (pd.DataFrame): Dataframe with features and target
        target_col (str): Name of the target column
        
    Returns:
        model: Trained model
        dict: Metrics
    """
    logger.info("Starting model training")
    
    # Separate features and target
    X = df.drop(columns=[target_col, 'SK_ID_CURR'])
    y = df[target_col]
    
    # Handle categorical variables (simple approach for baseline: drop or encode)
    # For baseline, let's just drop non-numeric columns to ensure it runs
    X_numeric = X.select_dtypes(include=['number'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
    
    # Pipeline: Impute -> Scale -> Model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.0001, max_iter=1000)) # Low C for regularization
    ])
    
    logger.info("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    logger.info(f"Model trained. Validation AUC: {auc:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "baseline_model.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return pipeline, {"auc": auc}
