import pandas as pd
import numpy as np
from src.utils import setup_logging
from src.features.bureau import join_bureau
from src.features.previous_application import join_previous_application
from src.features.pos_cash import join_pos_cash
from src.features.installments import join_installments
from src.features.credit_card import join_credit_card

logger = setup_logging()

def build_features(df: pd.DataFrame, bureau: pd.DataFrame = None, prev_app: pd.DataFrame = None, 
                   pos_cash: pd.DataFrame = None, installments: pd.DataFrame = None, 
                   credit_card: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate new features for the dataset using modularized functions.
    
    Args:
        df (pd.DataFrame): Dataframe with raw/cleaned features
        bureau (pd.DataFrame): Bureau data
        prev_app (pd.DataFrame): Previous application data
        pos_cash (pd.DataFrame): POS CASH balance data
        installments (pd.DataFrame): Installments payments data
        credit_card (pd.DataFrame): Credit card balance data
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    logger.info("Starting feature engineering")
    
    df_feat = df.copy()
    
    # Join auxiliary data if provided
    if bureau is not None:
        df_feat = join_bureau(df_feat, bureau)
        
    if prev_app is not None:
        df_feat = join_previous_application(df_feat, prev_app)

    if pos_cash is not None:
        df_feat = join_pos_cash(df_feat, pos_cash)
        
    if installments is not None:
        df_feat = join_installments(df_feat, installments)
        
    if credit_card is not None:
        df_feat = join_credit_card(df_feat, credit_card)
    
    # Domain knowledge features
    logger.info("Creating domain knowledge features")
    
    # Credit amount relative to income
    df_feat['CREDIT_INCOME_PERCENT'] = df_feat['AMT_CREDIT'] / df_feat['AMT_INCOME_TOTAL']
    
    # Annuity amount relative to income
    df_feat['ANNUITY_INCOME_PERCENT'] = df_feat['AMT_ANNUITY'] / df_feat['AMT_INCOME_TOTAL']
    
    # Credit term (approximate)
    df_feat['CREDIT_TERM'] = df_feat['AMT_ANNUITY'] / df_feat['AMT_CREDIT']
    
    # Days employed relative to age
    df_feat['DAYS_EMPLOYED_PERCENT'] = df_feat['DAYS_EMPLOYED'] / df_feat['DAYS_BIRTH']
    
    # Flag for anomalous days employed (365243 is often used as a placeholder)
    df_feat['DAYS_EMPLOYED_ANOM'] = df_feat['DAYS_EMPLOYED'] == 365243
    df_feat['DAYS_EMPLOYED'] = df_feat['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    logger.info(f"Feature engineering completed. New shape: {df_feat.shape}")
    return df_feat
