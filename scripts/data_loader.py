import os 
import sys
import pandas as pd

def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """
    Loads the dataset from the given file path.

    Args:
        file_path (str): Path to the dataset file (relative to project root or absolute).
        delimiter (str): The delimiter used in the file (default is comma).

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        # If path is not absolute, make it relative to project root
        if not os.path.isabs(file_path):
            file_path = os.path.join(get_project_root(), file_path)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        data = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Data loaded successfully from: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Project root: {get_project_root()}")
        return pd.DataFrame()
