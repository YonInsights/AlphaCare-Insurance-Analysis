from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from typing import Any, Dict

class InsuranceModel:
    def __init__(self, model_type: str):
        """
        Initialize the model with the specified type.
        
        Parameters:
        -----------
        model_type : str
            The type of model to use. Options are 'logistic', 'random_forest', and 'linear_regression'.
        """
        self.model_type = model_type
        self.model = self._get_model()
        self.feature_names = None

    def _get_model(self) -> Any:
        """
        Return the appropriate model based on the model_type.
        """
        if self.model_type == 'logistic':
            return LogisticRegression(random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                class_weight='balanced'
            )
        elif self.model_type == 'linear_regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: Any, y: Any) -> None:
        """
        Train the model using the given features and target variable.
        
        Parameters:
        -----------
        X : array-like
            The feature matrix.
        y : array-like
            The target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """
        Predict the target variable using the trained model.
        
        Parameters:
        -----------
        X : array-like
            The feature matrix to predict.
        """
        return self.model.predict(X)

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        X_test : array-like
            The test feature matrix.
        y_test : array-like
            The true target values for the test set.
        
        Returns:
        --------
        dict
            A dictionary of evaluation metrics.
        """
        y_pred = self.predict(X_test)
        
        if self.model_type in ['logistic', 'random_forest']:
            # Classification metrics
            y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
            return {
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
            }
        elif self.model_type == 'linear_regression':
            # Regression metrics
            return {
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }

def initialize_model(model_type: str = 'linear_regression') -> InsuranceModel:
    """
    Initialize the machine learning model based on the specified model type.
    
    Parameters:
    -----------
    model_type : str
        The type of model to initialize. Defaults to 'linear_regression'.
    
    Returns:
    --------
    model : InsuranceModel
        The initialized machine learning model.
    """
    return InsuranceModel(model_type=model_type)

def train_and_predict(model: InsuranceModel, X_train: Any, y_train: Any, X_test: Any) -> Dict[str, Any]:
    """
    Train the model and make predictions.
    
    Parameters:
    -----------
    model : InsuranceModel
        The machine learning model to train.
    X_train : array-like
        The training feature set.
    y_train : array-like
        The training target values.
    X_test : array-like
        The testing feature set.
        
    Returns:
    --------
    dict
        The model's evaluation metrics for the test set.
    """
    model.train(X_train, y_train)
    return model.evaluate(X_test, y_train)
