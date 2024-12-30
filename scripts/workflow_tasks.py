from insurance_model import initialize_model, train_and_predict
from statistical_model import evaluate_model
from visualizations import plot_predictions_vs_actuals
from data_preparation import prepare_data 

def run_insurance_model_workflow(data):
    """
    Execute the complete model workflow
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset
    
    Returns:
    --------
    dict
        Model performance metrics and visualization details
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data, target_column='your_target_column')
    
    # Initialize the model
    insurance_model = initialize_model('linear_regression')
    
    # Train the model and make predictions
    predictions = train_and_predict(insurance_model, X_train, y_train, X_test)
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, predictions, model_type='regression')
    
    # Visualize predictions
    plot_predictions_vs_actuals(y_test, predictions)
    
    return {
        'predictions': predictions,
        'metrics': evaluation_metrics
    }

if __name__ == '__main__':
    # Example usage
    import pandas as pd
    data = pd.read_csv('your_dataset.csv')
    results = run_insurance_model_workflow(data)
    print(results['metrics'])