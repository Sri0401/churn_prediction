import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Relative or absolute path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] Loaded data from: {file_path}")
        return data
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {file_path}")
        raise e
    except Exception as e:
        print(f"[ERROR] Failed to load data: {str(e)}")
        raise e