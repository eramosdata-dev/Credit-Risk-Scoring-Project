import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import sys
import argparse
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import setup_logging
from src.models.train import load_data
from src.features.build_features import build_features
from src.features.selection import remove_collinear_features

logger = setup_logging()

def objective(trial, data, target):
    # Define search space
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Fixed for tuning, rely on early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'bagging_freq': 1,
        'n_jobs': -1
    }
    
    # Stratified K-Fold for stable evaluation
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(data, target)):
        train_x, train_y = data.iloc[train_idx], target.iloc[train_idx]
        valid_x, valid_y = data.iloc[valid_idx], target.iloc[valid_idx]
        
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)
        
        # Pruning callback
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), pruning_callback]
        )
        
        # Predict
        preds = model.predict(valid_x, num_iteration=model.best_iteration)
        score = roc_auc_score(valid_y, preds)
        scores.append(score)
    
    return np.mean(scores)

def tune_model(data_dir, n_trials=20, debug=False):
    # Load and prepare data ONCE
    df, bureau, prev, pos, ins, cc = load_data(data_dir)
    
    if debug:
        logger.info("Debug mode: using subset of data")
        df = df.head(1000)
        bureau = bureau.head(1000)
        prev = prev.head(1000)
        pos = pos.head(1000)
        ins = ins.head(1000)
        cc = cc.head(1000)
        n_trials = 2 # Minimal trials for debug

    logger.info("Building features for tuning...")
    df_feat = build_features(df, bureau, prev, pos, ins, cc)
    df_feat = remove_collinear_features(df_feat, threshold=0.98)
    
    if 'TARGET' not in df_feat.columns:
        logger.error("TARGET column not found!")
        return

    train_df = df_feat[df_feat['TARGET'].notnull()]
    target = train_df['TARGET']
    data = train_df.drop(columns=['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'], errors='ignore')
    
    logger.info(f"Data shape for tuning: {data.shape}")
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data, target), n_trials=n_trials)
    
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value: {trial.value}")
    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best params
    output_dir = os.path.join(data_dir, '..', 'models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pd.DataFrame([trial.params]).to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)
    
    return trial.params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    tune_model(args.data_dir, n_trials=args.n_trials, debug=args.debug)
