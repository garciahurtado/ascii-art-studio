import inspect
import os
import types
import torch
from torch import nn
from datetime import datetime

models_path = os.path.normpath('models/')
def load_model(dataset, filename, num_labels):
    filename = os.path.join(models_path, dataset, filename)
    checkpoint = torch.load(filename + ".pt")

    # Load the source code of your custom model
    with open(filename + ".py", 'r') as f:
        model_code = f.read()

    # Create a module from the source code
    module_name = 'custom_model_module'
    model_module = types.ModuleType(module_name)
    exec(model_code, model_module.__dict__)

    # Get the class object from the module

    # Instantiate the model class
    model = model_module.AsciiClassifierNetwork(num_labels=num_labels)

    # Load the state dictionary
    model.load_state_dict(checkpoint.state_dict())

    return model

def save_model(model, model_filename):
    model_cass_filename = model_filename + ".py"
    checkpoint_filename = model_filename + ".pt"

    torch.save(model, checkpoint_filename)

    # Save the source code to a file so we can reload it later along with the weights
    filename = model.__module__
    filename = filename.replace(".", "\\") + ".py"
    with open(os.path.join(filename), 'r') as file:
        source_code = file.read()

    with open(model_cass_filename, 'w') as f:
        f.write(source_code)

def generate_model_filename(dataset):
    filename = datetime.now().strftime("%b%d_%H-%M-%S")
    model_file = os.path.join(models_path, dataset, f"{dataset}-{filename}")
    return model_file


