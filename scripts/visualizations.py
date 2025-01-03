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
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of numeric features across groups
    
    Parameters:
    - data: DataFrame containing the data
    - numeric_features: List of numeric features to plot
    - group_column: Column containing group labels
    - figsize: Figure size tuple (width, height)
    - save_path: Optional path to save the plot
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
    save_or_show_plot(save_path)

def plot_statistical_results(
    results: Dict[str, Dict],
    title: str = "Statistical Test Results",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot p-values and significance levels for statistical tests
    
    Parameters:
    - results: Dictionary containing test results
    - title: Title for the plot
    - figsize: Figure size tuple (width, height)
    - save_path: Optional path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Combine numeric and categorical results
    all_features = {**results.get('numeric', {}), **results.get('categorical', {})}
    
    # Extract features and p-values
    features = list(all_features.keys())
    p_values = [result.get('p_value', 1.0) for result in all_features.values()]
    
    # Create the plot
    bars = plt.bar(features, p_values)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (α=0.05)')
    
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('P-value')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Color bars based on significance
    for bar, p_value in zip(bars, p_values):
        bar.set_color('green' if p_value <= 0.05 else 'red')
    
    plt.tight_layout()
    save_or_show_plot(save_path)

def plot_predictions_vs_actuals(
    y_true, 
    y_pred, 
    title: str = 'Predictions vs Actuals', 
    save_path: Optional[str] = None
):
    """
    Plot a scatter plot of predictions vs actual values.
    
    Parameters:
    -----------
    y_true : pd.Series or np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted target values.
    title : str, optional
        Title of the plot
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    save_or_show_plot(save_path)

# Visualization
try:
    plt.figure(figsize=(15, 5))
    
    # Province distributions
    plt.subplot(1, 2, 1)
    plot_group_distributions(
        data=data,
        numeric_features=['TotalClaims'],
        group_column='Province'
    )
    
    # Gender distributions
    plt.subplot(1, 2, 2)
    plot_group_distributions(
        data=data,
        numeric_features=['TotalClaims'],
        group_column='Gender'
    )
    
    plt.tight_layout()
    plt.show()
    
    # Statistical results visualization
    plt.figure(figsize=(15, 5))
    
    # Province results
    plt.subplot(1, 2, 1)
    plot_statistical_results(
        results=province_results,
        title='Statistical Tests: Province Differences'
    )
    
    # Gender results
    plt.subplot(1, 2, 2)
    plot_statistical_results(
        results=gender_results,
        title='Statistical Tests: Gender Differences'
    )
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error in visualization: {str(e)}")
def plot_feature_importance(feature_importance_sorted, top_n=10):
    """
    Plots the top N most important features based on their coefficients.

    Parameters:
    - feature_importance_sorted: Sorted DataFrame of feature importance
    - top_n: Number of top features to display
    """
    # Plot the top N most important features
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_sorted['Feature'].head(top_n), feature_importance_sorted['Coefficient'].head(top_n))
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top
    plt.show()