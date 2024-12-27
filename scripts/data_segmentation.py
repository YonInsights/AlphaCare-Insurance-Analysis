import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

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

def create_test_control_groups(
    data: pd.DataFrame,
    feature: str,
    group_a_value: str,
    group_b_value: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create test and control groups based on feature values
    
    Parameters:
    - data: DataFrame containing the data
    - feature: Feature to split on (e.g., 'Province', 'Gender')
    - group_a_value: Value for control group
    - group_b_value: Value for test group
    
    Returns:
    - Tuple of (control_group, test_group)
    """
    control_group = data[data[feature] == group_a_value].copy()
    test_group = data[data[feature] == group_b_value].copy()
    return control_group, test_group

def check_numeric_balance(
    control_group: pd.DataFrame,
    test_group: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Check balance of numeric features between groups
    
    Parameters:
    - control_group: Control group DataFrame
    - test_group: Test group DataFrame
    - features: List of numeric features to check
    
    Returns:
    - DataFrame with balance metrics
    """
    balance_metrics = []
    
    for feature in features:
        # Calculate statistics
        control_mean = control_group[feature].mean()
        test_mean = test_group[feature].mean()
        control_std = control_group[feature].std()
        test_std = test_group[feature].std()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            control_group[feature].dropna(),
            test_group[feature].dropna(),
            equal_var=False
        )
        
        # Calculate standardized difference
        pooled_std = np.sqrt((control_std**2 + test_std**2) / 2)
        std_diff = (test_mean - control_mean) / pooled_std if pooled_std != 0 else 0
        
        balance_metrics.append({
            'feature': feature,
            'control_mean': control_mean,
            'test_mean': test_mean,
            'difference': test_mean - control_mean,
            'std_difference': std_diff,
            'p_value': p_value,
            'balanced': abs(std_diff) < 0.1 and p_value > 0.05
        })
    
    return pd.DataFrame(balance_metrics)

def check_categorical_balance(
    control_group: pd.DataFrame,
    test_group: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Check balance of categorical features between groups
    
    Parameters:
    - control_group: Control group DataFrame
    - test_group: Test group DataFrame
    - features: List of categorical features to check
    
    Returns:
    - DataFrame with balance metrics
    """
    balance_metrics = []
    
    for feature in features:
        # Calculate distributions
        control_dist = control_group[feature].value_counts(normalize=True)
        test_dist = test_group[feature].value_counts(normalize=True)
        
        # Perform chi-square test
        contingency = pd.crosstab(
            pd.concat([control_group[feature], test_group[feature]]),
            pd.Series(['Control']*len(control_group) + ['Test']*len(test_group))
        )
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        
        # Calculate maximum difference in proportions
        all_categories = set(control_dist.index) | set(test_dist.index)
        max_diff = max(
            abs(control_dist.get(cat, 0) - test_dist.get(cat, 0))
            for cat in all_categories
        )
        
        balance_metrics.append({
            'feature': feature,
            'max_proportion_diff': max_diff,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'balanced': max_diff < 0.1 and p_value > 0.05
        })
    
    return pd.DataFrame(balance_metrics)

def ensure_group_comparability(
    control_group: pd.DataFrame,
    test_group: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Ensure test and control groups are comparable across features
    
    Parameters:
    - control_group: Control group DataFrame
    - test_group: Test group DataFrame
    - numeric_features: List of numeric features to check
    - categorical_features: List of categorical features to check
    
    Returns:
    - Dictionary containing balance check results
    """
    numeric_balance = check_numeric_balance(
        control_group, test_group, numeric_features
    )
    categorical_balance = check_categorical_balance(
        control_group, test_group, categorical_features
    )
    
    return {
        'numeric_balance': numeric_balance,
        'categorical_balance': categorical_balance,
        'groups_comparable': (
            numeric_balance['balanced'].all() and 
            categorical_balance['balanced'].all()
        )
    }