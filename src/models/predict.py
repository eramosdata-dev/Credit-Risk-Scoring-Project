import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import argparse
import pickle
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import setup_logging
from src.features.build_features import build_features
from src.features.selection import remove_collinear_features

logger = setup_logging()

def load_test_data(data_dir):
    logger.info("Loading test data...")
    df = pd.read_csv(os.path.join(data_dir, 'application_test.csv'))
    bureau = pd.read_csv(os.path.join(data_dir, 'bureau.csv'))
    prev = pd.read_csv(os.path.join(data_dir, 'previous_application.csv'))
    pos = pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'))
    ins = pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'))
    cc = pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'))
    return df, bureau, prev, pos, ins, cc

def predict(data_dir, debug=False):
    # Load models
    models_path = os.path.join(data_dir, '..', 'models', 'models.pkl')
    if not os.path.exists(models_path):
        logger.error("Models not found! Run train.py first.")
        return
        
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
        
    # Load data
    df, bureau, prev, pos, ins, cc = load_test_data(data_dir)
    
    if debug:
        logger.info("Debug mode: using subset of data")
        df = df.head(1000)
        bureau = bureau.head(1000)
        prev = prev.head(1000)
        pos = pos.head(1000)
        ins = ins.head(1000)
        cc = cc.head(1000)
        
    # Feature Engineering
    logger.info("Building features for test set...")
    df_feat = build_features(df, bureau, prev, pos, ins, cc)
    
    # Feature Selection (Must match training!)
    # Note: Ideally we should save the list of selected features during training and reuse it here.
    # For now, we re-run collinear removal with same threshold, assuming deterministic behavior.
    df_feat = remove_collinear_features(df_feat, threshold=0.98)
    
    # Align features with model
    # Get features from the first model
    model_feats = models[0].feature_name_
    
    # Add missing features with 0
    for f in model_feats:
        if f not in df_feat.columns:
            df_feat[f] = 0
            
    # Select only model features
    test_x = df_feat[model_feats]
    
    logger.info(f"Test shape: {test_x.shape}")
    
    # Predict (Average over folds)
    preds = np.zeros(test_x.shape[0])
    for model in models:
        preds += model.predict_proba(test_x)[:, 1] / len(models)
        
    # Create submission
    submission = pd.DataFrame({
        'SK_ID_CURR': df['SK_ID_CURR'],
        'TARGET': preds
    })
    
    output_dir = os.path.join(data_dir, '..', 'models')
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"Submission saved to {submission_path}")
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    predict(args.data_dir, debug=args.debug)
