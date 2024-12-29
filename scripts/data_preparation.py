import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List, Union

def prepare_data(
    data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_variable: str
) -> Tuple[pd.DataFrame, Dict[str, Union[StandardScaler, LabelEncoder]]]:
    """
    Prepare data for statistical modeling by:
    1. Handling missing values
    2. Scaling numeric features
    3. Encoding categorical features
    """
    # Create a copy of the data
    df = data.copy()
    transformers = {}
    
    try:
        # Handle missing values
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
        df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
        
        # Scale numeric features
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        transformers['scaler'] = scaler
        
        # Encode categorical features
        for feature in categorical_features:
            if feature != target_variable:  # Don't encode target variable here
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
                transformers[f'encoder_{feature}'] = encoder
        
        logger.info("Data preparation completed successfully.")
        return df, transformers
    
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise

def split_features_target(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split data into features and target
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    feature_columns : List[str]
        List of feature column names
    target_column : str
        Name of target column
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Features DataFrame and target Series
    """
    X = data[feature_columns]
    y = data[target_column]
    return X, y
