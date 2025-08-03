import mlflow
import functools
import inspect

import torch

def ml_log_params(func):
    """
    Decorator that logs function parameters to MLflow.
    Logs all parameters passed to the decorated function as MLflow parameters.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function's parameter names and values
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Convert all parameters to strings for MLflow logging
        params = {k: str(v) for k, v in bound_args.arguments.items() 
                 if k != 'self'}  # Skip 'self' for class methods
        
        # Log parameters to MLflow
        if params:
            mlflow.log_params(params)
        
        # Call the original function
        return func(*args, **kwargs)
    
    return wrapper

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment(experiment_id='989653151241646983')
mlflow.set_experiment(experiment_id='0') # test experiment

