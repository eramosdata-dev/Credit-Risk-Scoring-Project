import sys
import sys
from src.data.load import load_all_data
from src.data.clean import clean_application_train
from src.utils import setup_logging

logger = setup_logging()

def main():
    logger.info("Starting Credit Risk Scoring Pipeline")
    
    try:
        # 1. Load Data
        data = load_all_data()
        train_df = data['application_train']
        bureau = data.get('bureau')
        prev_app = data.get('previous_application')
        pos_cash = data.get('POS_CASH_balance')
        installments = data.get('installments_payments')
        credit_card = data.get('credit_card_balance')
        
        # 2. Clean Data
        train_df_cleaned = clean_application_train(train_df)
        
        # 3. Feature Engineering
        from src.features.build_features import build_features
        train_df_fe = build_features(train_df_cleaned, bureau=bureau, prev_app=prev_app, 
                                     pos_cash=pos_cash, installments=installments, credit_card=credit_card)
        
        # 4. Modeling
        from src.models.lgbm_train import train_lightgbm
        model, metrics = train_lightgbm(train_df_fe)
        logger.info(f"Final Metrics: {metrics}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
