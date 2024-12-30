import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Data preparation function
def prepare_data(
    data: pd.DataFrame, 
    numeric_features: List[str], 
    categorical_features: List[str], 
    target_variable: str,
    date_features: List[str] = []  # New parameter to handle date columns
) -> Tuple[np.ndarray, pd.Series, Dict, List[str]]:
    """
    Prepare the data for model training by handling missing values, encoding categorical variables,
    scaling numeric variables, and converting date features.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    numeric_features (List[str]): List of column names that are numeric.
    categorical_features (List[str]): List of column names that are categorical.
    target_variable (str): The name of the target variable column.
    date_features (List[str]): List of column names that are date features (optional).

    Returns:
    Tuple[np.ndarray, pd.Series, Dict, List[str]]: Processed features (X), target variable (y),
    transformers (scalers/encoders), and feature names after preprocessing.
    """
    # Validate columns
    if not all(col in data.columns for col in numeric_features + categorical_features + [target_variable] + date_features):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    logging.info("Starting data preparation...")

    # Handle missing values for numeric and categorical features
    logging.info("Filling missing values in numeric features using the mean.")
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

    logging.info("Filling missing values in categorical features using the mode.")
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    # Convert date features to numerical values (e.g., timestamps)
    if date_features:
        logging.info("Converting date features to numerical values (timestamps).")
        for col in date_features:
            data[col] = pd.to_datetime(data[col], errors='coerce')
            data[col] = data[col].astype(np.int64) // 10**9  # Convert to seconds since epoch

    # Split data into features (X) and target (y)
    X = data[numeric_features + categorical_features + date_features]
    y = data[target_variable]

    # Create transformers for preprocessing (imputation, scaling, encoding)
    transformers = {
        'numeric': Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]),
        'categorical': Pipeline(steps=[ 
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
    }

    # Apply preprocessing to numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformers['numeric'], numeric_features),
            ('cat', transformers['categorical'], categorical_features)
        ]
    )

    # Apply the transformations
    logging.info("Applying transformations to numeric and categorical features.")
    X_processed = preprocessor.fit_transform(X)

    # Extract feature names after transformation
    feature_names = list(preprocessor.get_feature_names_out())

    logging.info("Data preparation complete.")
    return X_processed, y, transformers, feature_names


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
    # Validate columns
    if not all(col in data.columns for col in feature_columns + [target_column]):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    logging.info("Splitting data into features and target.")
    X = data[feature_columns]
    y = data[target_column]
    return X, y
