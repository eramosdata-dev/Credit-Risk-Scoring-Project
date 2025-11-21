# Credit Risk Scoring Project

This project implements a credit risk scoring model using the Home Credit Default Risk dataset.

## Project Structure

The project has been restructured into a professional Python package:

```
credit_risk_scoring_project/
├── src/
│   ├── data/               # Data loading and cleaning
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and evaluation
│   ├── config.py           # Configuration
│   └── utils.py            # Utilities
├── notebooks/              # Exploratory Data Analysis
├── data/                   # Data directory (raw and processed)
├── main.py                 # Entry point for the pipeline
└── README.md               # This file
```

## How to Run

1.  Ensure you have the required dependencies installed (pandas, scikit-learn, joblib).
2.  Run the main pipeline:
    ```bash
    python main.py
    ```
    This will:
    - Load data from `data/raw/application_train.csv`.
    - Clean the data.
    - Generate features.
    - Train a baseline Logistic Regression model.
    - Save the model to `models/baseline_model.joblib`.
    - Print validation metrics.

## Notebooks

- `01_exploracion_y_limpieza.ipynb`: Initial data exploration.
- `02_feature_engineering.ipynb`: Feature engineering exploration.

## Next Steps

- Expand feature engineering with more domain-specific features.
- Implement more advanced models (LightGBM, XGBoost).
- Add unit tests in `tests/`.
