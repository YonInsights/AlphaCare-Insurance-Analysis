import os
import sys
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
