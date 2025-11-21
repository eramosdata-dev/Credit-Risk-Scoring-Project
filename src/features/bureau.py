import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging()

def join_bureau(df: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Join bureau data to the main dataframe with advanced features.
    """
    logger.info("Joining bureau data...")
    
    # --- 1. Preprocessing ---
    # Active loans vs Closed loans
    bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau['CREDIT_ENDDATE_BINARY'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
    
    # --- 2. General Aggregations ---
    bureau_agg_dict = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'CREDIT_ACTIVE_BINARY': ['mean'], # % of active loans
        'CREDIT_ENDDATE_BINARY': ['mean'] # % of loans with future end date
    }
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_agg_dict)
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # --- 3. Active Loans Aggregations ---
    active_bureau = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = active_bureau.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'DAYS_CREDIT': ['mean', 'min'],
        'DAYS_CREDIT_ENDDATE': ['max', 'mean']
    })
    active_agg.columns = pd.Index(['BUREAU_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.merge(active_agg, on='SK_ID_CURR', how='left')

    # --- 4. Time-based Aggregations (Last 6, 12 months) ---
    # DAYS_CREDIT is negative (days before current application)
    for months in [6, 12]:
        days = months * 30
        time_df = bureau[bureau['DAYS_CREDIT'] >= -days]
        time_agg = time_df.groupby('SK_ID_CURR').agg({
            'SK_ID_BUREAU': ['count'],
            'AMT_CREDIT_SUM': ['sum'],
            'AMT_CREDIT_MAX_OVERDUE': ['max']
        })
        time_agg.columns = pd.Index([f'BUREAU_LAST_{months}M_' + e[0] + "_" + e[1].upper() for e in time_agg.columns.tolist()])
        bureau_agg = bureau_agg.merge(time_agg, on='SK_ID_CURR', how='left')

    # --- 5. Ratios ---
    # Debt / Credit Limit ratio (approximate)
    bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
    
    # Count of past loans
    bureau_agg['BUREAU_LOAN_COUNT'] = bureau.groupby('SK_ID_CURR').size()
    
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    return df
