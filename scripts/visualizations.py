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
def plot_categorical_distribution(data, column_name):
    """Plots a bar chart for categorical variable distribution."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column_name, palette="Set2")
    plt.title(f'Distribution of {column_name}')
    plt.xticks(rotation=45)
    plt.show()
def plot_scatter(data, x_column, y_column, title, x_label, y_label):
    """Plots a scatter plot to visualize the relationship between two numerical columns."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x=x_column, y=y_column, alpha=0.6)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

