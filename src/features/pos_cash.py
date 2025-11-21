import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def join_pos_cash(df: pd.DataFrame, pos_cash: pd.DataFrame) -> pd.DataFrame:
    """
    Join POS_CASH_balance data to the main dataframe with advanced features.
    """
    logger.info("Joining POS_CASH_balance data...")
    
    # --- 1. Preprocessing ---
    # Late payment flag
    pos_cash['LATE_PAYMENT'] = (pos_cash['SK_DPD'] > 0).astype(int)
    
    # --- 2. General Aggregations ---
    pos_agg_dict = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean'],
        'LATE_PAYMENT': ['mean']
    }
    
    pos_agg = pos_cash.groupby('SK_ID_CURR').agg(pos_agg_dict)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # --- 3. Recent Status (Last Month) ---
    # MONTHS_BALANCE is negative, -1 is last month
    last_month = pos_cash[pos_cash['MONTHS_BALANCE'] == -1]
    last_month_agg = last_month.groupby('SK_ID_CURR').agg({
        'SK_DPD': ['max'],
        'NAME_CONTRACT_STATUS': ['first'] # Status of most recent contract
    })
    
    # Handle categorical if needed, but for now let's stick to numericals or simple flags
    # If we want contract status, we need to encode it.
    # Let's skip categorical for now to avoid dimensionality explosion without careful selection
    
    # --- 4. Count of active loans ---
    pos_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()

    df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
    return df
