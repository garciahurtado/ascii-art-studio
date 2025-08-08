from urllib.parse import urlparse

import mlflow
import functools
import inspect

import requests
import torch
import logging

from const import INK_GREEN
from debugger import printc

logger = logging.getLogger("mlflow")

# Set the desired log level
# For example, to set the log level to DEBUG:
logger.setLevel(logging.DEBUG)


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


def is_server_running(tracking_uri=None):
    """
    Checks if the MLflow tracking server is running at the specified URI.

    Args:
        tracking_uri (str, optional): The URI of the MLflow tracking server.
                                      If None, it uses the currently set URI.

    Returns:
        bool: True if the server is reachable and responds, False otherwise.
    """
    if tracking_uri is None:
        tracking_uri = mlflow.get_tracking_uri()
    try:
        # A simple and quick check, a GET request to the root URL
        # Set a low timeout to fail fast if the server is down
        response = requests.get(tracking_uri, timeout=3)

        # Check for a successful response status code (e.g., 200 OK)
        if response.status_code == 200:
            print(f"MLflow server found at {tracking_uri}", INK_GREEN)
            return True
        else:
            printc(f"MLflow server at {tracking_uri} returned status code {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        printc(f"Failed to connect to MLflow server at {tracking_uri}: {e}")
        return False


def start_run():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if is_server_running():
        # Configure the client, now that we know the server is running
        mlflow.set_experiment(experiment_id='989653151241646983')  # ASCII Vision - c64
        mlflow.start_run()
    else:
        raise ConnectionError("MLflow server is not running.")
