import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_control_test_groups(
    data: pd.DataFrame,
    feature: str,
    test_size: float = 0.5,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into control and test groups
    
    Parameters:
    - data: DataFrame containing the data
    - feature: Feature to split on
    - test_size: Proportion of data to include in test group
    - random_state: Random seed for reproducibility
    
    Returns:
    - Tuple of (control_group, test_group)
    """
    control_group, test_group = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[feature] if feature in data.columns else None
    )
    return control_group, test_group

def check_group_balance(
    control_group: pd.DataFrame,
    test_group: pd.DataFrame,
    features: list
) -> pd.DataFrame:
    """
    Check if control and test groups are balanced across features
    
    Parameters:
    - control_group: Control group DataFrame
    - test_group: Test group DataFrame
    - features: List of features to check
    
    Returns:
    - DataFrame with balance metrics
    """
    balance_metrics = []
    
    for feature in features:
        if control_group[feature].dtype in ['int64', 'float64']:
            # Numerical features
            control_mean = control_group[feature].mean()
            test_mean = test_group[feature].mean()
            difference = abs(control_mean - test_mean)
            
            balance_metrics.append({
                'feature': feature,
                'control_metric': control_mean,
                'test_metric': test_mean,
                'difference': difference,
                'type': 'numerical'
            })
        else:
            # Categorical features
            control_dist = control_group[feature].value_counts(normalize=True)
            test_dist = test_group[feature].value_counts(normalize=True)
            max_diff = max(abs(control_dist - test_dist))
            
            balance_metrics.append({
                'feature': feature,
                'control_metric': 'distribution',
                'test_metric': 'distribution',
                'difference': max_diff,
                'type': 'categorical'
            })
    
    return pd.DataFrame(balance_metrics)

def ensure_group_comparability(
    control_group: pd.DataFrame,
    test_group: pd.DataFrame,
    features: list,
    threshold: float = 0.1
) -> bool:
    """
    Ensure that control and test groups are comparable
    
    Parameters:
    - control_group: Control group DataFrame
    - test_group: Test group DataFrame
    - features: List of features to check
    - threshold: Maximum allowed difference between groups
    
    Returns:
    - Boolean indicating if groups are comparable
    """
    balance_metrics = check_group_balance(control_group, test_group, features)
    return all(balance_metrics['difference'] < threshold)