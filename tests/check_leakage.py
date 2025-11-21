import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import build_features

def check_data_leakage():
    print("Checking for data leakage...")
    
    # 1. Create dummy data with a known target
    df = pd.DataFrame({
        'SK_ID_CURR': range(100),
        'TARGET': np.random.randint(0, 2, 100),
        'AMT_INCOME_TOTAL': np.random.rand(100) * 100000,
        'AMT_CREDIT': np.random.rand(100) * 300000,
        'AMT_ANNUITY': np.random.rand(100) * 15000,
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, 100),
        'DAYS_BIRTH': np.random.randint(-20000, -10000, 100)
    })
    
    # Generate features
    print("Generating features...")
    df_feat = build_features(df)
    
    # 2. Check for ID columns in features
    id_cols = [col for col in df_feat.columns if 'SK_ID' in col or 'ID' in col]
    # SK_ID_CURR is expected as key, but others might be suspicious if used as features
    print(f"ID columns found: {id_cols}")
    
    # 3. Check for perfect correlation with TARGET
    # (excluding TARGET itself)
    if 'TARGET' in df_feat.columns:
        corrs = df_feat.corr()['TARGET'].abs().sort_values(ascending=False)
        print("\nTop correlations with TARGET:")
        print(corrs.head(10))
        
        suspicious = corrs[corrs > 0.99]
        suspicious = suspicious.drop('TARGET', errors='ignore')
        
        if not suspicious.empty:
            print(f"\n[WARNING] Found suspicious features with correlation > 0.99: {suspicious.index.tolist()}")
            print("Possible leakage!")
        else:
            print("\n[PASS] No features with perfect correlation found.")
    else:
        print("\n[WARNING] TARGET column not found in output. Cannot check correlation.")

    # 4. Check for future information (heuristic)
    # We can't easily check this automatically without knowing the semantics, 
    # but we can check if any 'future' sounding columns exist.
    future_keywords = ['future', 'next', 'subsequent']
    future_cols = [col for col in df_feat.columns if any(keyword in col.lower() for keyword in future_keywords)]
    
    if future_cols:
        print(f"\n[WARNING] Found columns with future-related keywords: {future_cols}")
        print("Please verify these are not leakage.")
    else:
        print("\n[PASS] No obvious future-related column names found.")

if __name__ == "__main__":
    check_data_leakage()
