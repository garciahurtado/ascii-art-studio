import inspect
import os
import shutil
import sys
import types
import torch
from torch import nn
from datetime import datetime

from charset import Charset
from const import INK_BLUE, DATASETS_ROOT, MODELS_ROOT
from debugger import printc


def load_model(dataset, filename, num_labels):
    """
    Loads a Pytorch model from two files: the checkpoint file with the model state dictionary, and the source code
    of the model class, by the same name as the checkpoint, but with a .py extension.
    """
    base_path = get_full_model_path(dataset, filename)
    model_filename = base_path + ".pt"
    source_filename = base_path + ".py"

    printc(f"Loading Pytorch model from: {model_filename}", INK_BLUE)
    printc(f"Pytorch version: {torch.__version__}", INK_BLUE)
    printc(f"Python version: {sys.version}", INK_BLUE)
    printc(f"CUDA version: {torch.version.cuda}", INK_BLUE)
    printc(f"Models root: {MODELS_ROOT}", INK_BLUE)
    printc(f"Datasets / sources root: {DATASETS_ROOT}", INK_BLUE)

    # Load the source code of our custom model, by the same name as the checkpoint
    with open(source_filename, 'r') as f:
        model_code = f.read()

    # Create a module object from the model code and load / run the source code
    module_name = 'model_class_module'
    model_module = types.ModuleType(module_name)
    exec(model_code, model_module.__dict__)

    # Instantiate the model class (@TODO: this class name needs to be made dynamic)
    model = model_module.AsciiC64Network(num_labels=num_labels)
    # model = model_module.AsciiClassifierNetwork(num_labels=num_labels)

    # Load the state dictionary
    checkpoint = torch.load(model_filename, weights_only=False)

    if type(checkpoint) is dict:  # backwards compatibility for old checkpoint (object returned) vs new checkpoint (dict returned)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # it's an object
        model.load_state_dict(checkpoint.state_dict())

    return model


def save_model_source(source_filename, dataset, target_dir):
    target_filename = os.path.join(target_dir, dataset + ".py")
    shutil.copyfile(source_filename, target_filename)

def make_model_directory(model_name: str, dataset: str, base_dir: str = None) -> str:
    """
    Don't ever call this function from any of the code in this module, instead
    pass the saved model directory as a parameter to those functions. It should only be called
    once per training run
    """
    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    if base_dir is None:
        base_dir = MODELS_ROOT

    full_dir = os.path.join(base_dir, dataset, f"{model_name}-{dataset}-{timestamp}")

    if os.path.exists(full_dir):
        printc(f"ERROR: model directory already exists: {full_dir}")
        exit(1)

    os.makedirs(full_dir)
    print(f"Created model directory: {full_dir}")

    return full_dir


def get_full_model_path(dataset, filename):
    """
    Returns the full path to the model file, minus the extension
    """
    model_dir = os.path.realpath(os.path.join(MODELS_ROOT, dataset, filename))

    return model_dir

def save_checkpoint(
    model, 
    optimizer, 
    epoch: int, 
    metrics: dict, 
    model_dir: str, 
    max_keep: int = 3
) -> str:
    """
    Save training checkpoint with model state, optimizer state, and metrics.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        model_dir: Directory to save checkpoints
        max_keep: Maximum number of checkpoints to keep (deletes oldest)
    
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Clean up old checkpoints
    if max_keep > 0:
        checkpoints = sorted([f for f in os.listdir(model_dir) if f.startswith('checkpoint_')])
        while len(checkpoints) > max_keep:
            os.remove(os.path.join(model_dir, checkpoints[0]))
            checkpoints = checkpoints[1:]
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        
    Returns:
        Dictionary containing epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', '')
    }

def check_all_paths():
    # 1. check that the models directory exists
    if not os.path.exists(MODELS_ROOT):
        print(f"ERROR: models directory does not exist: {MODELS_ROOT}")
        exit(1)

    # 2. check that the code directory exists
