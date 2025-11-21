import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.selection import remove_collinear_features, select_important_features

def test_collinearity():
    print("\nTesting collinearity removal...")
    # Create dummy data
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10], # Perfectly correlated with A
        'C': [5, 4, 3, 2, 1], # Perfectly negatively correlated with A
        'D': [1, 5, 2, 6, 3]  # Random
    })
    
    print("Original shape:", df.shape)
    
    # Test removal
    df_clean = remove_collinear_features(df, threshold=0.95)
    
    print("Cleaned shape:", df_clean.shape)
    print("Remaining columns:", df_clean.columns.tolist())
    
    if 'B' not in df_clean.columns and 'C' not in df_clean.columns:
        print("[PASS] Removed correlated columns B and C")
    else:
        print("[FAIL] Did not remove all correlated columns")
        
def test_importance():
    print("\nTesting importance selection...")
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    # Informative features
    X1 = np.random.rand(n_samples)
    X2 = np.random.rand(n_samples)
    
    # Noise features
    X3 = np.random.rand(n_samples)
    X4 = np.random.rand(n_samples)
    
    # Target depends on X1 and X2
    y = (X1 + X2 > 1).astype(int)
    
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
    
    print("Original columns:", df.columns.tolist())
    
    # Test selection
    df_selected = select_important_features(df, pd.Series(y), n_features=2)
    
    print("Selected columns:", df_selected.columns.tolist())
    
    if 'X1' in df_selected.columns and 'X2' in df_selected.columns:
        print("[PASS] Selected informative features X1 and X2")
    else:
        print("[FAIL] Did not select informative features")

if __name__ == "__main__":
    test_collinearity()
    test_importance()
