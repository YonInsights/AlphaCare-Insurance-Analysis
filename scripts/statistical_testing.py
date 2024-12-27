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
    """
    try:
        # Create contingency table
        contingency_table = pd.crosstab(data[feature], data[group_column])
        
        # Check if we have enough data
        if contingency_table.size == 0:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'degrees_of_freedom': 0,
                'significant': False,
                'error': 'Insufficient data for chi-square test'
            }
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'test_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'degrees_of_freedom': 0,
            'significant': False,
            'error': str(e)
        }

def perform_t_test(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    equal_var: bool = False
) -> Dict[str, Union[float, bool]]:
    """
    Perform independent t-test for numerical variables
    """
    try:
        # Remove NaN values
        group1_clean = group1_data[~np.isnan(group1_data)]
        group2_clean = group2_data[~np.isnan(group2_data)]
        
        # Check if we have enough data
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data for t-test'
            }
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            group1_clean,
            group2_clean,
            equal_var=equal_var,
            nan_policy='omit'
        )
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': str(e)
        }

def perform_z_test(
    group1_data: np.ndarray,
    group2_data: np.ndarray
) -> Dict[str, Union[float, bool]]:
    """
    Perform z-test for large samples (n > 30)
    """
    try:
        # Remove NaN values
        group1_clean = group1_data[~np.isnan(group1_data)]
        group2_clean = group2_data[~np.isnan(group2_data)]
        
        # Check if we have enough data
        if len(group1_clean) < 30 or len(group2_clean) < 30:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data for z-test (n < 30)'
            }
        
        # Calculate means and standard errors
        mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
        se1, se2 = stats.sem(group1_clean), stats.sem(group2_clean)
        
        # Calculate z-statistic
        z_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
        
        # Calculate two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'test_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': str(e)
        }

def analyze_group_differences(
    data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    group_column: str
) -> Dict[str, Dict]:
    """
    Analyze differences between groups for both numeric and categorical features
    """
    try:
        results = {'numeric': {}, 'categorical': {}}
        
        # Check if group column exists
        if group_column not in data.columns:
            raise ValueError(f"Group column '{group_column}' not found in data")
        
        # Get unique groups
        unique_groups = data[group_column].unique()
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups in '{group_column}' for comparison")
        
        # Analyze numeric features
        for feature in numeric_features:
            if feature not in data.columns:
                results['numeric'][feature] = {
                    'error': f"Feature '{feature}' not found in data"
                }
                continue
                
            groups = data.groupby(group_column)[feature]
            group_sizes = groups.size()
            
            # Get data for first two groups
            group1_data = data[data[group_column] == unique_groups[0]][feature].values
            group2_data = data[data[group_column] == unique_groups[1]][feature].values
            
            # Choose appropriate test based on sample size
            if all(group_sizes > 30):
                test_results = perform_z_test(group1_data, group2_data)
                test_type = 'z_test'
            else:
                test_results = perform_t_test(group1_data, group2_data)
                test_type = 't_test'
                
            results['numeric'][feature] = {
                'test_type': test_type,
                **test_results
            }
        
        # Analyze categorical features
        for feature in categorical_features:
            if feature not in data.columns:
                results['categorical'][feature] = {
                    'error': f"Feature '{feature}' not found in data"
                }
                continue
                
            if feature == group_column:
                continue
                
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
        
    except Exception as e:
        return {
            'error': str(e),
            'numeric': {},
            'categorical': {}
        }