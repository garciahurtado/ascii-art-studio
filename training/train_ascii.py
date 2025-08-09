import math
import os
import inspect

import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import mlflow as ml

from charset import Charset
from const import INK_GREEN, INK_BLUE
from datasets.ascii_c64.ascii_c64_dataset import AsciiC64Dataset
from debugger import printc
from net.ascii_c64_network import AsciiC64Network
from performance_monitor import PerformanceMonitor
import datasets.data_utils as data_utils
import pytorch.model_manager as models
import mlflow_utils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def train_model(num_labels, dataset_class, charset):
    # Eliminate randomness to increase training reproducibility
    torch_rnd_seed = np_rnd_seed = 123456
    torch.manual_seed(torch_rnd_seed)
    np.random.seed(np_rnd_seed)
    mlflow_utils.start_run()

    char_width = charset.char_width
    char_height = charset.char_height
    dataset_name = dataset_class.dataset_name

    # Define training parameters, including the new augmentation flag
    train_params = {
        'batch_size': 1024,
        'test_batch_size': 1024 * 8,
        'train_test_split': 0.8,  # This is purely for logging, not functional
        'num_epochs': 1,
        'num_labels': num_labels,
        'learning_rate': 0.001,
        'decay_rate': 0.98,
        'decay_every_samples': 64000,
        'test_every_steps': 64,
        'log_every': 4,
        'augment_training_data': False  # Master switch for augmentation
    }

    # Calculate derived parameters
    train_params['decay_every_steps'] = math.ceil(train_params['decay_every_samples'] / train_params['batch_size'])
    # train_params['decay_every_samples'] = math.ceil(train_params['decay_every_steps'] * train_params['batch_size'])

    augment_params = None
    # Conditionally log augmentation parameters if enabled
    if train_params['augment_training_data']:
        # augment_params = {
        #     "RandomRotation_degrees": 0,
        #     "RandomAffine_translate_x": 0.1,
        #     "RandomAffine_translate_y": 0.1,
        #     "RandomAffine_fill": 0,
        #     "RandomHorizontalFlip_p": 0.5
        # }
        augment_params = {
            "RandomRotation_degrees": 0,
            "RandomAffine_translate_x": 0.5,
            "RandomAffine_translate_y": 0,
            "RandomAffine_fill": 0,
            "RandomHorizontalFlip_p": 0
        }
        ml.log_params(augment_params)

    # Load datasets
    trainset = data_utils.get_dataset(
        subdir='processed/train',
        dataset_class=dataset_class,
        char_width=char_width,
        char_height=char_height)

    testset = data_utils.get_dataset(
        subdir='processed/test',
        dataset_class=dataset_class,
        char_width=char_width,
        char_height=char_height)

    trainset.load_metadata()
    meta = trainset.metadata

    ml.log_artifact(trainset.metadata_path)

    # log basic dataset config
    ml.log_params({
        'torch_rnd_seed': torch_rnd_seed,
        'np_rnd_seed': np_rnd_seed,
        'dataset_name': dataset_name,
        'dataset_class': dataset_class.__name__,
        'dataset_version': trainset.metadata['version'] if trainset.metadata else trainset.version,
        'dataset_created_on': trainset.metadata['created'],
        'dataset_num_classes': meta['num_classes'],
        'dataset_train_count': len(trainset),
        'dataset_test_count': len(testset),
        'char_width': char_width,
        'char_height': char_height,
        'charset_filename': charset.filename,
        'charset_count': len(charset.chars),
        'charset_inverted': charset.inverted_included,
    })

    charset_file = str(os.path.join(charset.CHARSETS_DIR, charset.filename))
    ml.log_artifact(charset_file)   # Save the charset image that was used during training data generation

    class_counts = trainset.get_class_counts()
    num_train_samples = len(trainset)
    train_params['num_train_samples'] = num_train_samples
    train_params['steps_per_epoch'] = num_train_samples / train_params['batch_size']

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=3,
        prefetch_factor=1,
        drop_last=True,
        worker_init_fn=data_utils.seed_init_fn)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=train_params['test_batch_size'],
        shuffle=True,
        num_workers=3,
        prefetch_factor=1,
        drop_last=True,
        worker_init_fn=data_utils.seed_init_fn)

    train(class_counts, trainloader, testloader, train_params, dataset_name, charset)


def train(class_counts, trainloader, testloader, train_params, dataset_name, charset: Charset):
    model_family = "ascii-vision"
    log_every = train_params['log_every']
    device = data_utils.get_device()

    print(f"Training started on device: {device}")

    # Get class counts
    class_cardinality = len(class_counts)
    print(f"Found {class_cardinality} classes in dataset (from class_counts)")
    print(f"Charset {charset.filename} has {len(charset.chars)} chars.")

    # This needs to be made configurable ASAP
    model = AsciiC64Network(num_labels=class_cardinality)
    source_class_name = model.__class__.__name__
    source_class_file = inspect.getfile(model.__class__)
    model.to(device)

    # Alongside the model weights, we also save the current version of the model source code
    model_dir = models.make_model_directory(model_family, dataset_name)
    model_filename = os.path.join(model_dir, f"{dataset_name}.pt")

    models.save_model_source(source_class_file, dataset_name, model_dir)

    # model_filename = models.get_full_model_dir(dataset_name, model_dir)
    printc(f"Trained weights will be saved to: {model_filename}", INK_BLUE)

    # Log hyperparameters
    ml.log_params(train_params)

    ml.log_params({
        'model_family': model_family,
        'model_path': model_dir,
        'source_class_name': source_class_name,
        'source_class_file': source_class_file
    })
    ml.log_artifact(source_class_file)  # Save the model source code

    # Initialize parameter weights:
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    model.apply(weights_init_uniform_rule)

    # Generate class weights automatically
    class_weights = data_utils.create_class_weights(class_counts, mu=0.002)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    # Set up the learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, train_params['decay_rate'])

    # Set up Tensorboard writer
    # Add image embeddings
    # images, labels = select_n_random(trainset.data, trainset.targets, 1000)
    # add_image_embeddings(writer, images, labels, classes)

    perf = PerformanceMonitor()
    test_perf = PerformanceMonitor()

    testloader_gen = iter(testloader)

    for epoch in range(train_params['num_epochs']):
        print()
        print(f"======== EPOCH {epoch}/{train_params['num_epochs']} ========")
        print()
        steps_per_epoch = len(trainloader)

        # Create a progress bar
        progress = tqdm(enumerate(trainloader, 0), total=steps_per_epoch, colour='green')

        """ Training loop """
        for step, [inputs, targets] in progress:
            model.train()
            perf.reset()

            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = criterion(output, targets)
            perf.loss.append(loss.item())

            # zero out the parameter gradients and run the backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get the class indices from logits
            perf.add_predictions(output, targets)

            global_step = (epoch * steps_per_epoch) + step

            # +1 below makes sure we don't decay on step 0
            if ((global_step + 1) % train_params['decay_every_steps']) == 0:
                # decrease the learning rate
                scheduler.step()

            # Is it time to log?
            if (global_step % log_every) == 0:
                ml.log_metrics({
                    "train/loss": loss.item(),
                    "train/accuracy": perf.get_accuracy(),
                    "learning_rate": scheduler.get_last_lr()[0]
                }, step=global_step)

            if ((global_step + 1) % train_params['test_every_steps']) == 0:
                # perform validation using test dataset
                test_accuracy = 0
                test_perf.reset()

                try:
                    test_input, test_targets = next(testloader_gen)
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    testloader_gen = iter(testloader)
                    test_input, test_targets = next(testloader_gen)

                test_input, test_targets = test_input.to(device), test_targets.to(device)
                output = model(test_input)

                test_loss = criterion(output, test_targets)
                test_perf.loss.append(test_loss.item())

                test_perf.add_predictions(output, test_targets)
                test_accuracy = test_perf.get_accuracy()

                ml.log_metrics({
                    "test/loss": test_loss,
                    "test/accuracy": test_accuracy
                }, step=global_step)

            avg_loss = perf.get_avg_loss()
            desc = f"Step: {step} / Loss: "
            desc += f"{avg_loss:.3f}" if avg_loss else "n/a"
            desc += f" / Accy: {perf.get_accuracy():.3f} "
            progress.set_description(desc)

            # end for each step in epoch

        """
        End of epoch - save checkpoint
        These metrics will be saved by PyTorch alongside the model, which is why they are duplicated below
        """
        metrics = {
            'train_accuracy': perf.get_accuracy(),
            'test_accuracy': test_accuracy if 'test_accuracy' in locals() else 0.0,
            'test_loss': test_loss.item() if 'test_loss' in locals() else float('inf'),
            'learning_rate': scheduler.get_last_lr()[0],
        }

        checkpoint_path = models.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            model_dir=model_dir,
            max_keep=3  # Keep only 3 most recent checkpoints
        )

        print(f"Checkpoint saved to {checkpoint_path}")
        printc("== End of Epoch ==", INK_GREEN)

        # Log epoch metrics
        ml.log_metrics({
            "epoch/train_accuracy": metrics['train_accuracy'],
            "epoch/test_accuracy": metrics['test_accuracy'],
            "epoch/test_loss": metrics['test_loss'],
            "epoch/learning_rate": metrics['learning_rate']
        }, step=global_step)

    # End of training - save final model
    printc('================= TRAINING FINISHED =================', INK_GREEN)

    full_model_name = f"{model_family}-{dataset_name}"

    # Save the final model weights to disk
    final_model_path = os.path.join(model_dir, f"{full_model_name}.pt")
    torch.save({
        'epoch': train_params['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, final_model_path)

    # Save the final model weights to MLFlow
    ml.log_artifact(final_model_path)

    # Log the final model to MLFlow
    ml.pytorch.log_model(
        pytorch_model=model,
        name=model_family,
        registered_model_name=full_model_name,
        pip_requirements=[f"torch=={torch.__version__}"]
    )


def weights_init_uniform_rule(model):
    classname = model.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = model.in_features
        y = 1.0 / np.sqrt(n)
        model.weight.data.uniform_(-y, y)
        model.bias.data.fill_(0)

def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def validate(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    device = data_utils.get_device()

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.float32).mean()

    return test_loss, accuracy


def calc_accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Get the accuracy with respect to the most likely label
    @ref https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002

    :param model:
    :param x:
    :param y:
    :return:
    """
    # get the scores for each class (or logits)
    # y_logits = model(x)  # unnormalized probs

    # return the values & indices with the largest value in the dimension where the scores for each class is
    # get the scores with largest values & their corresponding idx (so the class that is most likely)
    max_scores, max_idx_class = model(x).max(
        dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    n = x.size(0)
    assert (n == max_idx_class.size(0))
    # calulate acc (note .item() to do float division)
    acc = (max_idx_class == y).sum().item() / n
    return acc


if __name__ == "__main__":
    # Basic config  --------------------------------------
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, "../")

    dataset_class = AsciiC64Dataset
    charset_name = 'c64.png'

    # ----------------------------------------------------
    charset = Charset()
    charset.load(charset_name)
    char_width, char_height = charset.char_width, charset.char_height

    # Output charset details
    print(f"Charset Loaded: {charset_name}")
    print(f"  char width: {char_width}")
    print(f"  char height: {char_height}")
    print(f"  count: {len(charset.chars)}")

    num_classes = len(charset.chars) * 2
    dataset_name = dataset_class.dataset_name


    train_model(num_classes, dataset_class, charset)
