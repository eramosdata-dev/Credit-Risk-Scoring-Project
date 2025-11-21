import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def join_installments(df: pd.DataFrame, installments: pd.DataFrame) -> pd.DataFrame:
    """
    Join installments_payments data to the main dataframe with advanced features.
    """
    logger.info("Joining installments_payments data...")
    
    # --- 1. Preprocessing ---
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Flag for late payment
    installments['LATE_PAYMENT'] = (installments['DPD'] > 0).astype(int)
    
    # --- 2. General Aggregations ---
    ins_agg_dict = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'var'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
        'LATE_PAYMENT': ['mean', 'sum'] # % of late payments, total late payments
    }

    ins_agg = installments.groupby('SK_ID_CURR').agg(ins_agg_dict)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # --- 3. Time-based Aggregations (Last 1 year) ---
    # DAYS_INSTALMENT is negative. Last 1 year = > -365
    last_year = installments[installments['DAYS_INSTALMENT'] > -365]
    last_year_agg = last_year.groupby('SK_ID_CURR').agg({
        'DPD': ['sum', 'mean', 'max'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['sum']
    })
    last_year_agg.columns = pd.Index(['INSTAL_LAST_1Y_' + e[0] + "_" + e[1].upper() for e in last_year_agg.columns.tolist()])
    ins_agg = ins_agg.merge(last_year_agg, on='SK_ID_CURR', how='left')

    # --- 4. Trends (Simple) ---
    # Compare last 6 months vs last 12 months (average payment amount)
    # If last 6m > last 12m, payments are increasing? Or maybe debt is increasing.
    
    # Let's try to capture if DPD is increasing.
    # We can't easily do complex regression here without slowing things down too much, 
    # but we can compare the first 5 installments vs last 5 installments for each user.
    # Sorting by DAYS_INSTALMENT
    
    # Count of installments
    ins_agg['INSTAL_COUNT'] = installments.groupby('SK_ID_CURR').size()
    
    df = df.merge(ins_agg, on='SK_ID_CURR', how='left')
    return df
