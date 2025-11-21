import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import os
import sys
import argparse
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import setup_logging
from src.features.build_features import build_features
from src.features.selection import remove_collinear_features

logger = setup_logging()

def load_data(data_dir):
    logger.info("Loading data...")
    df = pd.read_csv(os.path.join(data_dir, 'application_train.csv'))
    bureau = pd.read_csv(os.path.join(data_dir, 'bureau.csv'))
    prev = pd.read_csv(os.path.join(data_dir, 'previous_application.csv'))
    pos = pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'))
    ins = pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'))
    cc = pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'))
    return df, bureau, prev, pos, ins, cc

def train_model(data_dir, n_folds=5, debug=False):
    # Load data
    df, bureau, prev, pos, ins, cc = load_data(data_dir)
    
    if debug:
        logger.info("Debug mode: using subset of data")
        df = df.head(1000)
        bureau = bureau.head(1000)
        prev = prev.head(1000)
        pos = pos.head(1000)
        ins = ins.head(1000)
        cc = cc.head(1000)

    # Feature Engineering
    logger.info("Building features...")
    df_feat = build_features(df, bureau, prev, pos, ins, cc)
    
    # Feature Selection (Basic cleaning)
    # Remove collinear features
    df_feat = remove_collinear_features(df_feat, threshold=0.98)
    
    # Prepare data for training
    if 'TARGET' not in df_feat.columns:
        logger.error("TARGET column not found!")
        return
        
    train_df = df_feat[df_feat['TARGET'].notnull()]
    
    logger.info(f"Training shape: {train_df.shape}")
    
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    
    # Load best params if available
    params_path = os.path.join(data_dir, '..', 'models', 'best_params.csv')
    if os.path.exists(params_path):
        logger.info("Loading best params...")
        best_params = pd.read_csv(params_path).iloc[0].to_dict()
        # Ensure integer params are int
        for k in ['num_leaves', 'max_depth', 'min_child_samples', 'n_estimators']:
            if k in best_params:
                best_params[k] = int(best_params[k])
        
        # Add static params
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
        best_params['verbosity'] = -1
        best_params['boosting_type'] = 'gbdt'
        best_params['n_jobs'] = -1
        params = best_params
    else:
        logger.info("Using default params...")
        params = {
            'nthread': -1,
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 39.3259775,
            'silent': -1,
            'verbose': -1, 
            'objective': 'binary',
            'metric': 'auc'
        }
        
    # Stratified K-Fold
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    scores = []
    models = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(**params)
        
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric='auc', callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        
        fold_score = roc_auc_score(valid_y, oof_preds[valid_idx])
        scores.append(fold_score)
        logger.info(f'Fold {n_fold+1} AUC: {fold_score:.6f}')
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        models.append(clf)
        
        # Clean up
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    logger.info(f'Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    logger.info(f'Mean AUC score %.6f' % np.mean(scores))
    
    # Output directory
    output_dir = os.path.join(data_dir, '..', 'models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save feature importance
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Save OOF predictions
    oof_df = pd.DataFrame({'SK_ID_CURR': train_df['SK_ID_CURR'], 'TARGET': train_df['TARGET'], 'PREDICTION': oof_preds})
    oof_df.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)
    
    # Save models (saving the last one or all? Let's save the last one for simplicity or a list)
    # For production, we might want to save all and average predictions, or retrain on full data.
    # Let's save the first fold model as a representative for now, or save all in a pickle.
    with open(os.path.join(output_dir, 'models.pkl'), 'wb') as f:
        pickle.dump(models, f)
        
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    train_model(args.data_dir, debug=args.debug)
