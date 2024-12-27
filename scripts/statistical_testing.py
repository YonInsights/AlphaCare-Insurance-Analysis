import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Optional

def calculate_risk_ratio(claims: float, premium: float) -> float:
    """Calculate risk ratio (claims/premium)"""
    return claims/premium if premium != 0 else np.nan

def conduct_ttest(group_a: pd.Series, group_b: pd.Series) -> Tuple[float, float]:
    """
    Conduct independent t-test between two groups
    
    Parameters:
    - group_a: First group's data
    - group_b: Second group's data
    
    Returns:
    - t_statistic: T-test statistic
    - p_value: P-value from the test
    """
    t_statistic, p_value = stats.ttest_ind(
        group_a.dropna(),
        group_b.dropna(),
        equal_var=False  # Using Welch's t-test which doesn't assume equal variances
    )
    return t_statistic, p_value

def conduct_chi_square(group_a: pd.Series, group_b: pd.Series) -> Tuple[float, float]:
    """
    Conduct chi-square test between two categorical groups
    
    Parameters:
    - group_a: First group's categorical data
    - group_b: Second group's categorical data
    
    Returns:
    - chi2_statistic: Chi-square test statistic
    - p_value: P-value from the test
    """
    # Create contingency table
    contingency = pd.crosstab(pd.concat([group_a, group_b]), 
                             pd.Series(['A']*len(group_a) + ['B']*len(group_b)))
    
    chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    return chi2_statistic, p_value

def analyze_risk_differences(
    data: pd.DataFrame,
    grouping_column: str,
    claims_column: str = 'TotalClaims',
    premium_column: str = 'TotalPremium'
) -> pd.DataFrame:
    """
    Analyze risk differences between groups
    
    Parameters:
    - data: DataFrame containing the data
    - grouping_column: Column to group by (e.g., 'Province', 'PostalCode')
    - claims_column: Column containing claims data
    - premium_column: Column containing premium data
    
    Returns:
    - DataFrame with risk analysis results
    """
    results = data.groupby(grouping_column).agg({
        claims_column: ['sum', 'mean'],
        premium_column: ['sum', 'mean']
    }).reset_index()
    
    # Calculate risk ratio
    results['risk_ratio'] = results[(claims_column, 'sum')] / results[(premium_column, 'sum')]
    
    # Calculate profit margin
    results['profit_margin'] = (results[(premium_column, 'sum')] - results[(claims_column, 'sum')]) / results[(premium_column, 'sum')]
    
    return results

def compare_groups(
    data: pd.DataFrame,
    group_column: str,
    metric_column: str,
    group_a_value: str,
    group_b_value: str,
    test_type: str = 'ttest'
) -> Tuple[float, float, dict]:
    """
    Compare two groups using statistical tests
    
    Parameters:
    - data: DataFrame containing the data
    - group_column: Column used for grouping
    - metric_column: Column containing the metric to compare
    - group_a_value: Value identifying group A
    - group_b_value: Value identifying group B
    - test_type: Type of test to conduct ('ttest' or 'chi_square')
    
    Returns:
    - test_statistic: Test statistic
    - p_value: P-value from the test
    - summary: Dictionary containing summary statistics
    """
    group_a = data[data[group_column] == group_a_value][metric_column]
    group_b = data[data[group_column] == group_b_value][metric_column]
    
    summary = {
        'group_a_mean': group_a.mean(),
        'group_b_mean': group_b.mean(),
        'group_a_std': group_a.std(),
        'group_b_std': group_b.std(),
        'group_a_size': len(group_a),
        'group_b_size': len(group_b)
    }
    
    if test_type == 'ttest':
        test_statistic, p_value = conduct_ttest(group_a, group_b)
    else:
        test_statistic, p_value = conduct_chi_square(group_a, group_b)
    
    return test_statistic, p_value, summary
