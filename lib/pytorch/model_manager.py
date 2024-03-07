import os
from datetime import datetime

import torch
from ascii import NeuralAsciiConverterPytorch


models_path = os.path.normpath('models/')
def load_model(dataset, filename):
    filename = os.path.join(models_path, dataset, filename)
    model = torch.load(filename)
    return model

def generate_model_filename(dataset):
    filename = datetime.now().strftime("%b%d_%H-%M-%S")
    model_file = os.path.join(models_path, dataset, f"{dataset}-{filename}.pt")
    return model_file