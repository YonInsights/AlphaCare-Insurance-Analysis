import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_model_comparison_plot(results, save_path):
    """Create and save model comparison plot"""
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    r2_scores = [results[model]['r2_score'] for model in models]
    
    plt.bar(models, r2_scores, color=['#2ecc71', '#3498db'])
    plt.title('Model Performance Comparison (R² Score)')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.savefig(os.path.join(save_path, 'model_comparison.png'))
    plt.close()

def create_feature_importance_plot(importance_df, save_path, top_n=10):
    """Create and save feature importance plot"""
    plt.figure(figsize=(12, 6))
    
    # Plot top N features
    top_features = importance_df.head(top_n)
    sns.barplot(x='Coefficient', y='Feature', data=top_features)
    
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

def create_prediction_scatter_plot(y_true, y_pred, save_path):
    """Create and save prediction vs actual scatter plot"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_scatter.png'))
    plt.close()

def create_residuals_plot(y_true, y_pred, save_path):
    """Create and save residuals plot"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'residuals.png'))
    plt.close()

def generate_all_visualizations(results, feature_importance, y_true, y_pred):
    """Generate all visualizations for the report"""
    # Create reports directory if it doesn't exist
    reports_dir = "../reports/figures"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate all plots
    create_model_comparison_plot(results, reports_dir)
    create_feature_importance_plot(feature_importance, reports_dir)
    create_prediction_scatter_plot(y_true, y_pred, reports_dir)
    create_residuals_plot(y_true, y_pred, reports_dir)
    
    print(f"All visualizations have been saved to {reports_dir}")

if __name__ == "__main__":
    # Load results and generate visualizations
    # This section would be populated with actual results from your model runs
    pass
