import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def join_credit_card(df: pd.DataFrame, credit_card: pd.DataFrame) -> pd.DataFrame:
    """
    Join credit_card_balance data to the main dataframe with advanced features.
    """
    logger.info("Joining credit_card_balance data...")
    
    # --- 1. Preprocessing ---
    # Utilization Ratio
    credit_card['LIMIT_USE'] = credit_card['AMT_BALANCE'] / (credit_card['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    
    # Payment / Min Installment Ratio
    credit_card['PAYMENT_DIV_MIN'] = credit_card['AMT_PAYMENT_CURRENT'] / (credit_card['AMT_INST_MIN_REGULARITY'] + 1)
    
    # Late Payment
    credit_card['LATE_PAYMENT'] = (credit_card['SK_DPD'] > 0).astype(int)
    
    # --- 2. General Aggregations ---
    cc_agg_dict = {
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['min', 'max', 'mean', 'sum'],
        'LIMIT_USE': ['max', 'mean'],
        'PAYMENT_DIV_MIN': ['min', 'mean'],
        'LATE_PAYMENT': ['mean', 'sum']
    }
    
    cc_agg = credit_card.groupby('SK_ID_CURR').agg(cc_agg_dict)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    # --- 3. Count of credit card records ---
    cc_agg['CC_COUNT'] = credit_card.groupby('SK_ID_CURR').size()
    
    df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
    return df
