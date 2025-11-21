import sys
import os
import shutil
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train import train_model
from src.models.predict import predict
from tests.test_training import create_dummy_csvs

def test_prediction_pipeline():
    print("Testing prediction pipeline...")
    test_data_dir = 'tests/test_data_pred'
    create_dummy_csvs(test_data_dir)
    
    # Create application_test.csv (copy of train for simplicity, drop TARGET)
    train_df = pd.read_csv(os.path.join(test_data_dir, 'application_train.csv'))
    test_df = train_df.drop(columns=['TARGET'])
    test_df.to_csv(os.path.join(test_data_dir, 'application_test.csv'), index=False)
    
    try:
        print("Running train_model...")
        train_model(test_data_dir, n_folds=2, debug=False)
        
        print("Running predict...")
        submission = predict(test_data_dir, debug=False)
        
        if submission is not None and not submission.empty:
            print(f"[PASS] Prediction completed. Submission shape: {submission.shape}")
        else:
            print("[FAIL] Prediction failed.")
            
    except Exception as e:
        print(f"[FAIL] Exception: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
            
        # Cleanup models dir created in tests
        models_dir = os.path.join(test_data_dir, '..', 'models')
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)

if __name__ == "__main__":
    test_prediction_pipeline()
