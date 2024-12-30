from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

    def _get_model(self):
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

    def train(self, X, y):
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

    def predict(self, X):
        """
        Predict the target variable using the trained model.
        
        Parameters:
        -----------
        X : array-like
            The feature matrix to predict.
        """
        return self.model.predict(X)

def initialize_model(model_type='linear_regression'):
    """
    Initialize the machine learning model based on the specified model type.
    
    Parameters:
    -----------
    model_type : str
        The type of model to initialize. Defaults to 'linear_regression'.
    
    Returns:
    --------
    model : object
        The initialized machine learning model.
    """
    insurance_model = InsuranceModel(model_type)
    return insurance_model.model

def train_and_predict(model, X_train, y_train, X_test):
    """
    Train the model and make predictions.
    
    Parameters:
    -----------
    model : object
        The machine learning model to train.
    X_train : pd.DataFrame or np.ndarray
        The training feature set.
    y_train : pd.Series or np.ndarray
        The training target values.
    X_test : pd.DataFrame or np.ndarray
        The testing feature set.
        
    Returns:
    --------
    predictions : np.ndarray
        The model's predictions for the test set.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions
