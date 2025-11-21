import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def join_previous_application(df: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Join previous application data to the main dataframe with advanced features.
    """
    logger.info("Joining previous application data...")
    
    # --- 1. Preprocessing ---
    # Application status flags
    prev['APP_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['APP_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    # --- 2. General Aggregations ---
    prev_agg_dict = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'APP_APPROVED': ['mean', 'sum'],
        'APP_REFUSED': ['mean', 'sum']
    }
    
    prev_agg = prev.groupby('SK_ID_CURR').agg(prev_agg_dict)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # --- 3. Approved Applications Aggregations ---
    approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
    approved_agg = approved.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'mean'] # min days decision means most recent
    })
    approved_agg.columns = pd.Index(['PREV_APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.merge(approved_agg, on='SK_ID_CURR', how='left')
    
    # --- 4. Refused Applications Aggregations ---
    refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
    refused_agg = refused.groupby('SK_ID_CURR').agg({
        'AMT_APPLICATION': ['mean', 'max'],
        'DAYS_DECISION': ['min', 'mean']
    })
    refused_agg.columns = pd.Index(['PREV_REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.merge(refused_agg, on='SK_ID_CURR', how='left')

    # --- 5. Last Application Features (Most recent) ---
    # Sort by DAYS_DECISION (descending, so closest to 0 is first if negative, but DAYS_DECISION is usually negative)
    # Actually DAYS_DECISION is negative relative to current application. Max value is closest to current.
    last_app = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False]).groupby('SK_ID_CURR').first()
    
    # Select interesting columns from last application
    last_app_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION']
    last_app = last_app[last_app_cols]
    last_app.columns = ['PREV_LAST_' + c for c in last_app.columns]
    
    # One-hot encode categorical status for last app
    if 'PREV_LAST_NAME_CONTRACT_STATUS' in last_app.columns:
        last_app = pd.get_dummies(last_app, columns=['PREV_LAST_NAME_CONTRACT_STATUS'], prefix='PREV_LAST')

    prev_agg = prev_agg.merge(last_app, on='SK_ID_CURR', how='left')

    # --- 6. Ratios ---
    # Application / Credit ratio (did they get what they asked for?)
    prev_agg['PREV_APP_CREDIT_RATIO'] = prev_agg['PREV_AMT_APPLICATION_MEAN'] / (prev_agg['PREV_AMT_CREDIT_MEAN'] + 1)
    
    # Count of previous applications
    prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size()
    
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    return df
