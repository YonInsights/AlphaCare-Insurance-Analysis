import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
            Type of model to use ('logistic' or 'random_forest')
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
            cv=cv_folds, scoring='roc_auc'
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
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'feature_importance': self.get_feature_importance()
        }
    
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
