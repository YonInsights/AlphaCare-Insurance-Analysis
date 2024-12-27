import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Union

def perform_chi_square_test(
    data: pd.DataFrame,
    feature: str,
    group_column: str
) -> Dict[str, Union[float, bool]]:
    """
    Perform chi-square test of independence for categorical variables
    
    Parameters:
    - data: DataFrame containing the data
    - feature: Categorical feature to test
    - group_column: Column containing group labels (e.g., 'Province', 'Gender')
    
    Returns:
    - Dictionary containing test statistics and interpretation
    """
    contingency_table = pd.crosstab(data[feature], data[group_column])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'test_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05
    }

def perform_t_test(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    equal_var: bool = False
) -> Dict[str, Union[float, bool]]:
    """
    Perform independent t-test for numerical variables
    
    Parameters:
    - group1_data: Data from first group
    - group2_data: Data from second group
    - equal_var: Whether to assume equal variances
    
    Returns:
    - Dictionary containing test statistics and interpretation
    """
    t_stat, p_value = stats.ttest_ind(
        group1_data,
        group2_data,
        equal_var=equal_var
    )
    
    return {
        'test_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def perform_z_test(
    group1_data: np.ndarray,
    group2_data: np.ndarray
) -> Dict[str, Union[float, bool]]:
    """
    Perform z-test for large samples (n > 30)
    
    Parameters:
    - group1_data: Data from first group
    - group2_data: Data from second group
    
    Returns:
    - Dictionary containing test statistics and interpretation
    """
    # Calculate means and standard errors
    mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
    se1, se2 = stats.sem(group1_data), stats.sem(group2_data)
    
    # Calculate z-statistic
    z_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
    
    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'test_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def analyze_group_differences(
    data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    group_column: str
) -> Dict[str, Dict]:
    """
    Analyze differences between groups for both numeric and categorical features
    
    Parameters:
    - data: DataFrame containing the data
    - numeric_features: List of numeric features to analyze
    - categorical_features: List of categorical features to analyze
    - group_column: Column containing group labels
    
    Returns:
    - Dictionary containing test results for all features
    """
    results = {'numeric': {}, 'categorical': {}}
    
    # Analyze numeric features
    for feature in numeric_features:
        groups = data.groupby(group_column)[feature]
        group_sizes = groups.size()
        
        # Choose appropriate test based on sample size
        if all(group_sizes > 30):
            # Use Z-test for large samples
            test_results = perform_z_test(
                groups.get_group(groups.groups[0]).values,
                groups.get_group(groups.groups[1]).values
            )
            test_type = 'z_test'
        else:
            # Use T-test for smaller samples
            test_results = perform_t_test(
                groups.get_group(groups.groups[0]).values,
                groups.get_group(groups.groups[1]).values
            )
            test_type = 't_test'
            
        results['numeric'][feature] = {
            'test_type': test_type,
            **test_results
        }
    
    # Analyze categorical features
    for feature in categorical_features:
        test_results = perform_chi_square_test(
            data,
            feature,
            group_column
        )
        results['categorical'][feature] = {
            'test_type': 'chi_square',
            **test_results
        }
    
    return results