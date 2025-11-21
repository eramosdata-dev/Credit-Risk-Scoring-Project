import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tune import tune_model
from tests.test_training import create_dummy_csvs

def test_tuning_pipeline():
    print("Testing tuning pipeline...")
    test_data_dir = 'tests/test_data_tune'
    create_dummy_csvs(test_data_dir)
    
    try:
        print("Running tune_model...")
        # Run with very few trials and debug mode for speed
        best_params = tune_model(test_data_dir, n_trials=2, debug=True)
        
        if best_params:
            print(f"[PASS] Tuning completed successfully. Best params: {best_params}")
        else:
            print("[FAIL] Tuning failed or did not return params.")
            
    except Exception as e:
        print(f"[FAIL] Exception during tuning: {e}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)

if __name__ == "__main__":
    test_tuning_pipeline()
