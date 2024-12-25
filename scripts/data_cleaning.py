import os
import sys
import pandas as pd
def detect_outliers(df, column):
    """
    Function to detect outliers using the IQR method.
    
    Parameters:
    df (pandas.DataFrame): The dataset.
    column (str): The column name to detect outliers.
    
    Returns:
    pandas.DataFrame: DataFrame containing the outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def check_missing_values(df):
    """
    Function to check for missing values in the dataset.
    
    Parameters:
    df (pandas.DataFrame): The dataset to check.
    
    Returns:
    pandas.Series: A series containing the number of missing values per column.
    """
    return df.isnull().sum()
def remove_outliers(data, column):
    """
    Removes rows where the specified column contains outliers based on IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): The column name to check for outliers
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return cleaned_data

def calculate_monthly_changes(data, group_by_column, value_column):
    """
    Calculate the monthly change in a given value column, grouped by a specific column (e.g., 'PostalCode').
    
    Parameters:
    - data: DataFrame containing the data
    - group_by_column: Column to group by (e.g., 'PostalCode')
    - value_column: Column to calculate the monthly change for (e.g., 'TotalPremium', 'TotalClaims')
    
    Returns:
    - DataFrame with the original data and a new column for monthly changes
    """
    data = data.sort_values(by=[group_by_column, 'TransactionMonth'])
    data[f'{value_column}_Change'] = data.groupby(group_by_column)[value_column].diff().fillna(0)
    return data