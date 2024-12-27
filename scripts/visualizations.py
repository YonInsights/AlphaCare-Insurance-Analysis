from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from typing import Dict, List

# Plot configuration
DEFAULT_FIGURE_SIZE = (10, 6)
DEFAULT_DPI = 300
DEFAULT_CMAP = 'coolwarm'
PLOT_SAVE_KWARGS = {'bbox_inches': 'tight', 'dpi': DEFAULT_DPI}

def save_or_show_plot(save_path: Optional[str] = None) -> None:
    """Utility function to either save or display the current plot."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, **PLOT_SAVE_KWARGS)
        plt.close()
    else:
        plt.show()

def plot_boxplot(data: pd.DataFrame, column_name: str, save_path: Optional[str] = None) -> None:
    """
    This function generates a boxplot to visualize the distribution of a given column,
    highlighting the outliers.
    
    Parameters:
    - data (pd.DataFrame): The dataframe containing the data.
    - column_name (str): The name of the column to visualize.
    - save_path (str, optional): Path to save the plot. If None, plot is displayed instead.
    """
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    sns.boxplot(data=data, x=column_name, color='skyblue')
    plt.title(f'Boxplot of {column_name}')
    save_or_show_plot(save_path)

def plot_categorical_distribution(data: pd.DataFrame, column_name: str, save_path: Optional[str] = None) -> None:
    """
    Plots a bar chart for categorical variable distribution.
    
    Parameters:
    - data (pd.DataFrame): The dataframe containing the data.
    - column_name (str): The name of the column to visualize.
    - save_path (str, optional): Path to save the plot. If None, plot is displayed instead.
    """
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    sns.countplot(data=data, x=column_name, palette="Set2")
    plt.title(f'Distribution of {column_name}')
    plt.xticks(rotation=45)
    save_or_show_plot(save_path)

def plot_scatter(
    data: pd.DataFrame, 
    x_column: str, 
    y_column: str, 
    title: str, 
    x_label: str, 
    y_label: str, 
    geographic_column: Optional[str] = None, 
    palette: str = 'viridis', 
    save_path: Optional[str] = None
) -> None:
    """
    Plots a scatter plot to visualize the relationship between two numerical columns.
    Optionally, colors the points by a geographic or categorical column.
    
    Parameters:
    - data: DataFrame containing the data
    - x_column: The column for the x-axis
    - y_column: The column for the y-axis
    - title: Title of the plot
    - x_label: Label for the x-axis
    - y_label: Label for the y-axis
    - geographic_column: Optional column to color the points by (e.g., ZipCode or other categorical columns)
    - palette: Color palette for the scatter plot (default is 'viridis')
    - save_path (str, optional): Path to save the plot. If None, plot is displayed instead.
    """
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    
    if geographic_column:
        sns.scatterplot(data=data, x=x_column, y=y_column, hue=geographic_column, palette=palette, alpha=0.6)
    else:
        sns.scatterplot(data=data, x=x_column, y=y_column, alpha=0.6)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    save_or_show_plot(save_path)

def plot_geographic_trends(
    data: pd.DataFrame, 
    geographic_column: str, 
    comparison_column: str, 
    title: str, 
    x_label: str, 
    y_label: str, 
    top_n: int = 10, 
    save_path: Optional[str] = None
) -> None:
    """
    Plots trends over geographic regions comparing specified columns.
    
    Parameters:
    - data: DataFrame containing the data
    - geographic_column: Column containing geographic information
    - comparison_column: Column to compare across geographic regions
    - title: Title for the plot
    - x_label: Label for x-axis
    - y_label: Label for y-axis
    - top_n: Number of top regions to display
    - save_path (str, optional): Path to save the plot. If None, plot is displayed instead.
    """
    plt.figure(figsize=(12, 6))
    
    # Group by geographic column and calculate mean of comparison column
    grouped_data = data.groupby(geographic_column)[comparison_column].mean().sort_values(ascending=False).head(top_n)
    
    # Create bar plot
    grouped_data.plot(kind='bar')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    save_or_show_plot(save_path)

def plot_correlation_matrix(
    data: pd.DataFrame, 
    columns: List[str], 
    title: str = "Correlation Matrix", 
    save_path: Optional[str] = None
) -> None:
    """
    Plots the correlation matrix for the specified columns.
    
    Parameters:
    - data: DataFrame containing the data.
    - columns: List of columns to include in the correlation matrix.
    - title: Title for the plot.
    - save_path (str, optional): Path to save the plot. If None, plot is displayed instead.
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[columns].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap=DEFAULT_CMAP, center=0)
    plt.title(title)
    save_or_show_plot(save_path)
def plot_group_distributions(
    data: pd.DataFrame,
    numeric_features: List[str],
    group_column: str,
    figsize: tuple = (15, 5)
) -> None:
    """
    Plot distribution of numeric features across groups
    
    Parameters:
    - data: DataFrame containing the data
    - numeric_features: List of numeric features to plot
    - group_column: Column containing group labels
    - figsize: Figure size tuple (width, height)
    """
    n_features = len(numeric_features)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, numeric_features):
        sns.boxplot(data=data, x=group_column, y=feature, ax=ax)
        ax.set_title(f'{feature} by {group_column}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
def plot_statistical_results(
    results: Dict[str, Dict],
    title: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot p-values and significance levels for statistical tests
    
    Parameters:
    - results: Dictionary containing test results
    - title: Title for the plot
    - figsize: Figure size tuple (width, height)
    """
    # Combine numeric and categorical results
    all_features = {**results['numeric'], **results['categorical']}
    
    # Extract features and p-values
    features = list(all_features.keys())
    p_values = [results['p_value'] for results in all_features.values()]
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(features, p_values)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α=0.05)')
    
    # Color bars based on significance
    for bar, p_value in zip(bars, p_values):
        bar.set_color('green' if p_value < 0.05 else 'gray')
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('P-value')
    plt.legend()
    plt.tight_layout()