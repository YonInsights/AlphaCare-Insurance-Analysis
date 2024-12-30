import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    mean_squared_error, 
    r2_score
)
from typing import Dict, Tuple, Any, Union, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceModel:
    """Class to handle insurance modeling tasks"""
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize the model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic', 'random_forest', 'linear_regression', 'random_forest_regressor')
        """
        self.model_type = model_type
        self.model = self._get_model()
        self.feature_names = None
        
    def _get_model(self) -> Any:
        """Get the specified model instance"""
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                class_weight='balanced'
            )
        elif self.model_type == 'linear_regression':
            return LinearRegression()
        elif self.model_type == 'random_forest_regressor':
            return RandomForestRegressor(
                random_state=42,
                n_estimators=100
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train the model with cross-validation
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, float]
            Cross-validation scores
        """
        self.feature_names = X_train.columns
        
        # Train the model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv_folds, scoring='roc_auc' if self.model_type in ['logistic', 'random_forest'] else 'neg_mean_squared_error'
        )
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        Dict[str, Any]
            Evaluation metrics
        """
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model not trained yet!")
            
        y_pred = self.model.predict(X_test)
        evaluation_metrics = {}

        if self.model_type in ['logistic', 'random_forest']:
            y_prob = self.model.predict_proba(X_test)[:, 1]
            evaluation_metrics = {
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'feature_importance': self.get_feature_importance()
            }
        elif self.model_type in ['linear_regression', 'random_forest_regressor']:
            evaluation_metrics = {
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }

        return evaluation_metrics
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if self.feature_names is None:
            raise ValueError("Model not trained yet!")
            
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.Series()
        
        return pd.Series(
            importance,
            index=self.feature_names
        ).sort_values(ascending=False)

def prepare_insurance_data(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare insurance data for modeling
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target : str
        Target column name
    test_size : float
        Proportion of data for testing
    stratify : bool
        Whether to stratify the split
        
    Returns:
    --------
    Tuple
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing data for modeling...")
    
    # Validate input
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data!")
        
    # Split features and target
    X = data.drop(columns=[target])
    y = data[target]
    
    # Perform stratified split if requested
    stratify_data = y if stratify else None
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_data
    )

def evaluate_model(y_true, y_pred, model_type='classification'):
    """
    Evaluate the model performance using appropriate metrics.
    
    Parameters:
    -----------
    y_true : pd.Series or np.ndarray
        The true target values.
    y_pred : np.ndarray
        The predicted target values.
    model_type : str, optional
        Type of model evaluation. Options are 'classification' or 'regression'..
        
    Returns:
    --------
    dict
        A dictionary of evaluation metrics.
    """
    if model_type == 'classification':
        return {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
    elif model_type == 'regression':
        return {
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def extract_feature_importance(model, feature_names):
    """
    Extracts feature importance from the linear regression model.

    Parameters:
    - model: Trained linear regression model
    - feature_names: List of feature names

    Returns:
    - Sorted DataFrame of feature importance with coefficients
    """
    # Extract coefficients from the trained model
    coefficients = model.coef_

    # Match the coefficients with feature names
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort by absolute value of coefficients to rank features
    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance_sorted = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

    return feature_importance_sorted

