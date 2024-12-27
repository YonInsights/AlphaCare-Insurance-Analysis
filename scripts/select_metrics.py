import pandas as pd
import numpy as np
from typing import Dict, Any, List

def calculate_risk_ratio(claims: float, premium: float) -> float:
    """Calculate risk ratio (claims/premium)"""
    return claims/premium if premium != 0 else np.nan

def calculate_profit_margin(premium: float, claims: float) -> float:
    """Calculate profit margin ((premium-claims)/premium)"""
    return (premium - claims)/premium if premium != 0 else np.nan

class InsuranceMetrics:
    """Class to handle insurance-related metrics calculations"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with insurance data
        
        Parameters:
        - data: DataFrame containing insurance data with columns:
          TotalPremium, TotalClaims, Province, Gender, etc.
        """
        self.data = data
        
    def risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        return {
            'total_risk_ratio': calculate_risk_ratio(
                self.data['TotalClaims'].sum(),
                self.data['TotalPremium'].sum()
            ),
            'avg_risk_ratio': self.data.apply(
                lambda x: calculate_risk_ratio(x['TotalClaims'], x['TotalPremium']),
                axis=1
            ).mean(),
            'claims_frequency': len(self.data[self.data['TotalClaims'] > 0]) / len(self.data)
        }
    
    def profit_metrics(self) -> Dict[str, float]:
        """Calculate profit-related metrics"""
        return {
            'total_profit_margin': calculate_profit_margin(
                self.data['TotalPremium'].sum(),
                self.data['TotalClaims'].sum()
            ),
            'avg_profit_margin': self.data.apply(
                lambda x: calculate_profit_margin(x['TotalPremium'], x['TotalClaims']),
                axis=1
            ).mean(),
            'revenue_per_customer': self.data['TotalPremium'].mean()
        }
    
    def geographic_metrics(self, region_column: str) -> pd.DataFrame:
        """Calculate metrics by geographic region"""
        return self.data.groupby(region_column).agg({
            'TotalPremium': ['sum', 'mean'],
            'TotalClaims': ['sum', 'mean'],
            'PolicyID': 'count'
        }).round(2)
    
    def gender_metrics(self) -> pd.DataFrame:
        """Calculate metrics by gender"""
        return self.data.groupby('Gender').agg({
            'TotalPremium': ['sum', 'mean'],
            'TotalClaims': ['sum', 'mean'],
            'PolicyID': 'count'
        }).round(2)
    
    def get_all_kpis(self) -> Dict[str, Any]:
        """Get all KPIs in a single dictionary"""
        kpis = {
            'risk_metrics': self.risk_metrics(),
            'profit_metrics': self.profit_metrics(),
            'gender_metrics': self.gender_metrics().to_dict(),
            'province_metrics': self.geographic_metrics('Province').to_dict()
        }
        return kpis

def get_hypothesis_metrics() -> List[Dict[str, str]]:
    """Define metrics for each hypothesis"""
    return [
        {
            'hypothesis': 'Province Risk Differences',
            'primary_metric': 'risk_ratio',
            'description': 'Claims to Premium ratio by province',
            'business_impact': 'Identifies high-risk geographic areas'
        },
        {
            'hypothesis': 'Zip Code Risk Differences',
            'primary_metric': 'risk_ratio',
            'description': 'Claims to Premium ratio by postal code',
            'business_impact': 'Enables granular geographic risk assessment'
        },
        {
            'hypothesis': 'Zip Code Margin Differences',
            'primary_metric': 'profit_margin',
            'description': 'Profit margin by postal code',
            'business_impact': 'Identifies areas of high/low profitability'
        },
        {
            'hypothesis': 'Gender Risk Differences',
            'primary_metric': 'risk_ratio',
            'description': 'Claims to Premium ratio by gender',
            'business_impact': 'Assesses risk patterns across gender groups'
        }
    ]

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