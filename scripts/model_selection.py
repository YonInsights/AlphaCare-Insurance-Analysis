# model_selection.py
from insurance_model import InsuranceModel

def initialize_model(model_type: str) -> InsuranceModel:
    """
    Initialize the InsuranceModel based on the specified model type.

    Parameters:
    -----------
    model_type : str
        The type of model to use. Options are 'logistic', 'random_forest', and 'linear_regression'.

    Returns:
    --------
    InsuranceModel
        The initialized InsuranceModel object.
    """
    try:
        insurance_model = InsuranceModel(model_type=model_type)
        print(f"InsuranceModel initialized with {model_type} successfully.")
        return insurance_model
    except Exception as e:
        print(f"Error during model initialization: {e}")
        raise
