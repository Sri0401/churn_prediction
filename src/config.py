import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaned/preprocessed output path
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "src", "data", "cleaned_telco.csv")

# Model output path
MODEL_PATH = os.path.join(ROOT_DIR, "models", "rf_telco_churn.pkl")

# Evaluation results
EVAL_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")
