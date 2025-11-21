import pandas as pd
import numpy as np
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train import train_model

def create_dummy_csvs(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Create dummy files with minimal columns
    n_rows = 100
    
    # application_train
    df = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'TARGET': np.random.randint(0, 2, n_rows),
        'AMT_INCOME_TOTAL': np.random.rand(n_rows) * 100000,
        'AMT_CREDIT': np.random.rand(n_rows) * 300000,
        'AMT_ANNUITY': np.random.rand(n_rows) * 15000,
        'DAYS_EMPLOYED': np.random.randint(-10000, 0, n_rows),
        'DAYS_BIRTH': np.random.randint(-20000, -10000, n_rows)
    })
    df.to_csv(os.path.join(data_dir, 'application_train.csv'), index=False)
    
    # bureau
    bureau = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'SK_ID_BUREAU': range(n_rows),
        'DAYS_CREDIT': np.random.randint(-1000, 0, n_rows),
        'CREDIT_ACTIVE': ['Active'] * n_rows,
        'DAYS_CREDIT_ENDDATE': np.random.randint(-500, 500, n_rows),
        'DAYS_CREDIT_UPDATE': np.random.randint(-100, 0, n_rows),
        'CREDIT_DAY_OVERDUE': [0] * n_rows,
        'AMT_CREDIT_MAX_OVERDUE': [0] * n_rows,
        'AMT_CREDIT_SUM': np.random.rand(n_rows) * 10000,
        'AMT_CREDIT_SUM_DEBT': np.random.rand(n_rows) * 5000,
        'AMT_CREDIT_SUM_OVERDUE': [0] * n_rows,
        'AMT_CREDIT_SUM_LIMIT': [0] * n_rows,
        'CNT_CREDIT_PROLONG': [0] * n_rows
    })
    bureau.to_csv(os.path.join(data_dir, 'bureau.csv'), index=False)
    
    # Create other empty dummy files to satisfy load_data
    # Previous Application needs NAME_CONTRACT_STATUS and DAYS_DECISION for feature engineering
    prev_df = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'NAME_CONTRACT_STATUS': ['Approved'] * n_rows,
        'DAYS_DECISION': [-100] * n_rows,
        'AMT_ANNUITY': [0] * n_rows,
        'AMT_APPLICATION': [0] * n_rows,
        'AMT_CREDIT': [0] * n_rows,
        'AMT_DOWN_PAYMENT': [0] * n_rows,
        'AMT_GOODS_PRICE': [0] * n_rows,
        'HOUR_APPR_PROCESS_START': [0] * n_rows,
        'RATE_DOWN_PAYMENT': [0] * n_rows,
        'CNT_PAYMENT': [0] * n_rows
    })
    prev_df.to_csv(os.path.join(data_dir, 'previous_application.csv'), index=False)
    
    # POS CASH needs MONTHS_BALANCE and SK_DPD
    pos_df = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'MONTHS_BALANCE': [-1] * n_rows,
        'SK_DPD': [0] * n_rows,
        'SK_DPD_DEF': [0] * n_rows,
        'NAME_CONTRACT_STATUS': ['Active'] * n_rows
    })
    pos_df.to_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'), index=False)
    
    # Installments needs AMT_PAYMENT, AMT_INSTALMENT, DAYS_ENTRY_PAYMENT, DAYS_INSTALMENT
    ins_df = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'AMT_PAYMENT': [100] * n_rows,
        'AMT_INSTALMENT': [100] * n_rows,
        'DAYS_ENTRY_PAYMENT': [-10] * n_rows,
        'DAYS_INSTALMENT': [-10] * n_rows,
        'NUM_INSTALMENT_VERSION': [1] * n_rows
    })
    ins_df.to_csv(os.path.join(data_dir, 'installments_payments.csv'), index=False)
    
    # Credit Card needs AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL, etc.
    cc_df = pd.DataFrame({
        'SK_ID_CURR': range(n_rows),
        'MONTHS_BALANCE': [-1] * n_rows,
        'AMT_BALANCE': [0] * n_rows,
        'AMT_CREDIT_LIMIT_ACTUAL': [0] * n_rows,
        'AMT_DRAWINGS_ATM_CURRENT': [0] * n_rows,
        'AMT_DRAWINGS_CURRENT': [0] * n_rows,
        'AMT_DRAWINGS_OTHER_CURRENT': [0] * n_rows,
        'AMT_DRAWINGS_POS_CURRENT': [0] * n_rows,
        'AMT_INST_MIN_REGULARITY': [0] * n_rows,
        'AMT_PAYMENT_CURRENT': [0] * n_rows,
        'AMT_PAYMENT_TOTAL_CURRENT': [0] * n_rows,
        'AMT_RECEIVABLE_PRINCIPAL': [0] * n_rows,
        'AMT_RECIVABLE': [0] * n_rows,
        'AMT_TOTAL_RECEIVABLE': [0] * n_rows,
        'CNT_DRAWINGS_ATM_CURRENT': [0] * n_rows,
        'CNT_DRAWINGS_CURRENT': [0] * n_rows,
        'CNT_DRAWINGS_OTHER_CURRENT': [0] * n_rows,
        'CNT_DRAWINGS_POS_CURRENT': [0] * n_rows,
        'CNT_INSTALMENT_MATURE_CUM': [0] * n_rows,
        'SK_DPD': [0] * n_rows,
        'SK_DPD_DEF': [0] * n_rows
    })
    cc_df.to_csv(os.path.join(data_dir, 'credit_card_balance.csv'), index=False)

def test_training_pipeline():
    print("Testing training pipeline...")
    test_data_dir = 'tests/test_data'
    create_dummy_csvs(test_data_dir)
    
    try:
        print("Running train_model...")
        scores = train_model(test_data_dir, n_folds=2, debug=False)
        
        if scores and len(scores) == 2:
            print(f"[PASS] Training completed successfully. Scores: {scores}")
        else:
            print("[FAIL] Training failed or did not return scores.")
            
    except Exception as e:
        print(f"[FAIL] Exception during training: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)

if __name__ == "__main__":
    test_training_pipeline()
