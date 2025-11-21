import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import build_features

def create_dummy_data():
    # Main DF
    df = pd.DataFrame({
        'SK_ID_CURR': [100001, 100002, 100003],
        'AMT_INCOME_TOTAL': [100000, 200000, 150000],
        'AMT_CREDIT': [300000, 600000, 450000],
        'AMT_ANNUITY': [15000, 30000, 22500],
        'DAYS_EMPLOYED': [-1000, -2000, 365243],
        'DAYS_BIRTH': [-10000, -15000, -12000]
    })
    
    # Bureau
    bureau = pd.DataFrame({
        'SK_ID_CURR': [100001, 100001, 100002],
        'SK_ID_BUREAU': [1, 2, 3],
        'DAYS_CREDIT': [-100, -400, -50],
        'CREDIT_ACTIVE': ['Active', 'Closed', 'Active'],
        'DAYS_CREDIT_ENDDATE': [100, -100, 200],
        'DAYS_CREDIT_UPDATE': [-10, -50, -5],
        'CREDIT_DAY_OVERDUE': [0, 0, 0],
        'AMT_CREDIT_MAX_OVERDUE': [0, 0, 0],
        'AMT_CREDIT_SUM': [10000, 20000, 15000],
        'AMT_CREDIT_SUM_DEBT': [5000, 0, 10000],
        'AMT_CREDIT_SUM_OVERDUE': [0, 0, 0],
        'AMT_CREDIT_SUM_LIMIT': [0, 0, 0],
        'CNT_CREDIT_PROLONG': [0, 0, 0]
    })
    
    # Previous Application
    prev = pd.DataFrame({
        'SK_ID_CURR': [100001, 100002],
        'SK_ID_PREV': [10, 11],
        'NAME_CONTRACT_STATUS': ['Approved', 'Refused'],
        'AMT_ANNUITY': [5000, 0],
        'AMT_APPLICATION': [50000, 100000],
        'AMT_CREDIT': [50000, 0],
        'AMT_DOWN_PAYMENT': [0, 0],
        'AMT_GOODS_PRICE': [50000, 100000],
        'HOUR_APPR_PROCESS_START': [10, 12],
        'RATE_DOWN_PAYMENT': [0, 0],
        'DAYS_DECISION': [-500, -100],
        'CNT_PAYMENT': [12, 0]
    })
    
    return df, bureau, prev

def test_feature_generation():
    print("Creating dummy data...")
    df, bureau, prev = create_dummy_data()
    
    print("Running build_features...")
    df_feat = build_features(df, bureau=bureau, prev_app=prev)
    
    print("Verifying output...")
    print(f"Output shape: {df_feat.shape}")
    print(f"Columns: {df_feat.columns.tolist()}")
    
    # Check for new features
    expected_cols = [
        'BUREAU_DAYS_CREDIT_MEAN', 'BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM', 
        'BUREAU_LAST_6M_SK_ID_BUREAU_COUNT',
        'PREV_APPROVED_AMT_CREDIT_MEAN', 'PREV_LAST_AMT_CREDIT'
    ]
    
    for col in expected_cols:
        if col in df_feat.columns:
            print(f"[PASS] Found column: {col}")
        else:
            print(f"[FAIL] Missing column: {col}")
            
    print("Test completed.")

if __name__ == "__main__":
    test_feature_generation()
