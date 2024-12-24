# data_preprocessing.py
import pandas as pd

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        strategy (str): The strategy to handle missing values ('drop' or 'fill').
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        df = df.fillna(df.mean())  # You can modify this strategy as needed
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        strategy (str): The strategy to handle missing values ('drop' or 'fill').
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        df = df.fillna(df.mean())  # You can modify this strategy as needed
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return df

def convert_data_types(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """
    Converts columns in the DataFrame to the specified data types.
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        column_types (dict): A dictionary mapping column names to their target data types.
    Returns:
        pd.DataFrame: The DataFrame with the updated data types.
    """
    for column, dtype in column_types.items():
        if column in df.columns:
            try:
                df[column] = df[column].astype(dtype)
            except ValueError:
                print(f"Warning: Could not convert column '{column}' to {dtype}.")
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")
    
    return df
