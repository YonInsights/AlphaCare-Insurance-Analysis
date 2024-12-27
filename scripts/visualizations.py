from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

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