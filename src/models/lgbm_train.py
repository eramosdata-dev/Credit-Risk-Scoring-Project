import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.utils import setup_logging
from src.config import MODELS_DIR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logging()

def train_lightgbm(df: pd.DataFrame, target_col: str = 'TARGET', n_folds: int = 5):
    """
    Train a LightGBM model using Stratified K-Fold CV.
    
    Args:
        df (pd.DataFrame): Dataframe with features and target
        target_col (str): Name of the target column
        n_folds (int): Number of folds for CV
        
    Returns:
        model: Trained model (last fold or best iteration logic)
        dict: Metrics
    """
    logger.info("Starting LightGBM training with Stratified K-Fold")
    
    # Separate features and target
    X = df.drop(columns=[target_col, 'SK_ID_CURR'])
    y = df[target_col]
    
    # Handle categorical variables
    # LightGBM can handle categorical features directly if they are of type 'category'
    # But for safety/simplicity in this iteration, we'll label encode object columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
            
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    feature_importance_df = pd.DataFrame()
    
    scores = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1,
        )
        
        clf.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], 
            eval_metric='auc', 
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
        
        fold_score = roc_auc_score(y_valid, oof_preds[valid_idx])
        scores.append(fold_score)
        logger.info(f'Fold {n_fold + 1} AUC: {fold_score:.5f}')
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f'Overall AUC: {overall_auc:.5f}')
    
    # Save feature importance
    cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'lgbm_importances.png')
    
    # Save the last model (or we could retrain on full data)
    # For now, saving the last fold model as a representative
    model_path = MODELS_DIR / "lgbm_model.joblib"
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return clf, {"auc": overall_auc, "fold_scores": scores}
