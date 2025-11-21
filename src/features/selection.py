import pandas as pd
import numpy as np
import lightgbm as lgb
from src.utils import setup_logging

logger = setup_logging()

def remove_collinear_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove collinear features from the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with features
        threshold (float): Correlation threshold to remove features
        
    Returns:
        pd.DataFrame: Dataframe with collinear features removed
    """
    logger.info(f"Removing collinear features with threshold {threshold}...")
    
    # Calculate correlation matrix
    # Select only numerical columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info(f"Dropping {len(to_drop)} columns: {to_drop}")
    
    df_selected = df.drop(columns=to_drop)
    
    logger.info(f"Shape after collinearity removal: {df_selected.shape}")
    return df_selected

def select_important_features(df: pd.DataFrame, target: pd.Series, n_features: int = None, threshold: float = None) -> pd.DataFrame:
    """
    Select important features using LightGBM feature importance.
    
    Args:
        df (pd.DataFrame): Dataframe with features (should not include target)
        target (pd.Series): Target variable
        n_features (int): Number of top features to select
        threshold (float): Importance threshold (keep features with importance > threshold)
        
    Returns:
        pd.DataFrame: Dataframe with selected features
    """
    logger.info("Selecting important features using LightGBM...")
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    # Create LightGBM dataset
    dtrain = lgb.Dataset(df, label=target)
    
    # Train a simple model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(params, dtrain, num_boost_round=100)
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    selected_features = feature_names # Default to all
    
    if n_features:
        selected_features = feature_imp.head(n_features)['feature'].tolist()
        logger.info(f"Selected top {n_features} features.")
        
    elif threshold:
        selected_features = feature_imp[feature_imp['importance'] > threshold]['feature'].tolist()
        logger.info(f"Selected {len(selected_features)} features with importance > {threshold}.")
        
    logger.info(f"Top 10 features: {feature_imp.head(10)['feature'].tolist()}")
    
    return df[selected_features]
