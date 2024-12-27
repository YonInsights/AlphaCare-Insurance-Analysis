import pandas as pd
import numpy as np
from typing import Dict, Any
from visualizations import (
    plot_boxplot,
    plot_categorical_distribution,
    plot_scatter,
    plot_correlation_matrix
)

def generate_hypothesis_test_report(
    test_results: Dict[str, Any],
    hypothesis_name: str
) -> str:
    """
    Generate a formatted report for hypothesis test results
    
    Parameters:
    - test_results: Dictionary containing test results
    - hypothesis_name: Name of the hypothesis being tested
    
    Returns:
    - Formatted report string
    """
    report = f"""
    Hypothesis Test Results: {hypothesis_name}
    ----------------------------------------
    Test Type: {test_results['test_type']}
    Test Statistic: {test_results['statistic']:.4f}
    P-Value: {test_results['p_value']:.4f}
    
    Group A Mean: {test_results['group_a_mean']:.4f}
    Group B Mean: {test_results['group_b_mean']:.4f}
    Difference: {test_results['difference']:.4f}
    
    Conclusion: {
        'Reject null hypothesis' if test_results['significant']
        else 'Fail to reject null hypothesis'
    }
    """
    return report

def create_visualization_report(
    data: pd.DataFrame,
    test_results: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Create and save visualizations for the analysis
    
    Parameters:
    - data: DataFrame containing the data
    - test_results: Dictionary containing test results
    - output_dir: Directory to save visualizations
    """
    # Distribution plots
    plot_boxplot(
        data=data,
        column_name='TotalPremium',
        save_path=f'{output_dir}/premium_distribution.png'
    )
    
    # Categorical distributions
    plot_categorical_distribution(
        data=data,
        column_name='Gender',
        save_path=f'{output_dir}/gender_distribution.png'
    )
    
    # Scatter plots
    plot_scatter(
        data=data,
        x_column='TotalPremium',
        y_column='TotalClaims',
        title='Premium vs Claims',
        x_label='Total Premium',
        y_label='Total Claims',
        save_path=f'{output_dir}/premium_claims_scatter.png'
    )

def generate_full_report(
    data: pd.DataFrame,
    test_results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Generate a complete analysis report
    
    Parameters:
    - data: DataFrame containing the data
    - test_results: Dictionary containing all test results
    - output_dir: Directory to save outputs
    
    Returns:
    - Complete report string
    """
    report = "Insurance Analysis Report\n=====================\n\n"
    
    # Add each hypothesis test result
    for hypothesis_name, results in test_results.items():
        report += generate_hypothesis_test_report(results, hypothesis_name)
        report += "\n\n"
    
    # Create visualizations
    create_visualization_report(data, test_results, output_dir)
    
    return report