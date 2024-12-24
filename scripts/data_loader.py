import os 
import sys
import pandas as pd
def load_data(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """
    Loads the dataset from the given file path.

    Args:
        file_path (str): Path to the dataset file.
        delimiter (str): The delimiter used in the file (default is comma).

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(file_path, delimiter=delimiter)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
