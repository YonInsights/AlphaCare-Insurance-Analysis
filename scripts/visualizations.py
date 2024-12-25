# File: src/visualize_outliers.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(data, column_name):
    """
    This function generates a boxplot to visualize the distribution of a given column,
    highlighting the outliers.
    
    Parameters:
    - data (pd.DataFrame): The dataframe containing the data.
    - column_name (str): The name of the column to visualize.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=column_name, color='skyblue')
    plt.title(f'Boxplot of {column_name}')
    plt.show()
