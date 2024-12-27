import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_risk_ratio(claims: float, premium: float) -> float:
    """Calculate risk ratio (claims/premium)"""
    return claims/premium if premium != 0 else np.nan

def calculate_profit_margin(premium: float, claims: float) -> float:
    """Calculate profit margin ((premium-claims)/premium)"""
    return (premium - claims)/premium if premium != 0 else np.nan

def calculate_kpis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate key performance indicators for insurance data
    
    Parameters:
    - data: DataFrame containing insurance data
    
    Returns:
    - Dictionary containing calculated KPIs
    """
    kpis = {
        'total_premium': data['TotalPremium'].sum(),
        'total_claims': data['TotalClaims'].sum(),
        'average_premium': data['TotalPremium'].mean(),
        'average_claims': data['TotalClaims'].mean(),
        'risk_ratio': calculate_risk_ratio(
            data['TotalClaims'].sum(),
            data['TotalPremium'].sum()
        ),
        'profit_margin': calculate_profit_margin(
            data['TotalPremium'].sum(),
            data['TotalClaims'].sum()
        )
    }
    return kpis

def calculate_group_metrics(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Calculate metrics grouped by a specific column
    
    Parameters:
    - data: DataFrame containing insurance data
    - group_column: Column to group by (e.g., 'Province', 'Gender')
    
    Returns:
    - DataFrame with metrics for each group
    """
    metrics = data.groupby(group_column).agg({
        'TotalPremium': ['sum', 'mean'],
        'TotalClaims': ['sum', 'mean'],
        'PolicyID': 'count'
    }).reset_index()
    
    # Calculate risk ratio and profit margin
    metrics['risk_ratio'] = metrics[('TotalClaims', 'sum')] / metrics[('TotalPremium', 'sum')]
    metrics['profit_margin'] = (metrics[('TotalPremium', 'sum')] - metrics[('TotalClaims', 'sum')]) / metrics[('TotalPremium', 'sum')]
    
    return metrics