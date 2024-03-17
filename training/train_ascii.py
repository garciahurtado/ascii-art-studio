import math

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
import torchmetrics
import os
import sys
from datetime import datetime
from torchmetrics import Precision, Recall, F1Score

from torch.utils.data import Subset

import wandb

sys.path.append('../lib')

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from datasets.ascii_amstrad_cpc.ascii_amstrad_cpc import AsciiAmstradCPC
from datasets.ascii_c64.ascii_c64 import AsciiC64
from net.ascii_classifier_network import AsciiClassifierNetwork
from performance_monitor import PerformanceMonitor
import datasets.data_utils as data_utils
import pytorch.model_manager as models


def train_model(num_labels, dataset_type, dataset_name):
    # Eliminate randomness to increase training reproducibility
    torch.manual_seed(123456)
    np.random.seed(123456)
    random_state = 99

    batch_size = 2048
    test_batch_size = batch_size * 10

    # Load datasets
    trainset = data_utils.get_dataset(train=True, dataset_type=dataset_type, num_labels=num_labels)
    testset = data_utils.get_dataset(train=False, dataset_type=dataset_type, num_labels=num_labels)

    # trainset, testset = data_utils.split_dataset(trainset, 0.2, random_state=random_state, charset_name=dataset_name)
    class_counts = trainset.get_class_counts()
    num_train_samples = len(trainset)

    steps_per_epoch = num_train_samples / batch_size
    decay_every_samples = 512 * 1000

    params = {
        'batch_size': batch_size,
        'num_epochs': 30,
        'num_train_samples': num_train_samples,
        'steps_per_epoch': steps_per_epoch,
        'learning_rate': 0.005,
        'decay_rate': 0.96,
        'decay_every_steps': math.ceil(decay_every_samples / batch_size),
        'test_every_steps': 120,
        'log_every': 4,
    }

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size * 2,
        shuffle=True,
        num_workers=0,
        prefetch_factor=None,
        drop_last=True,
        worker_init_fn=data_utils.seed_init_fn)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0,
        prefetch_factor=None,
        drop_last=True,
        worker_init_fn=data_utils.seed_init_fn)

    train(class_counts, trainloader, testloader, params)


def train(class_counts, trainloader, testloader, params):
    global models_path
    log_every = params['log_every']
    device = data_utils.get_device()

    # Get class counts
    class_cardinality = len(class_counts)
    print(f"Found {class_cardinality} classes in dataset")

    model = AsciiClassifierNetwork(num_labels=class_cardinality)
    model.to(device)

    # Initialize parameter weights:
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    model.apply(weights_init_uniform_rule)

    # Generate class weights
    class_weights = data_utils.create_class_weights(class_counts, mu=0.0015)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights.to(device)

    # Loss function / optimizer / Learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, params['decay_rate'])

    # Logging
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key='e6533f84f309fe9d42fd1e3577f8ba162ed921c0')
    wandb.init(
        project='pytorch-c64',
        entity='ghurtado',
        config=params)

    dataset_name = 'ascii_c64'
    model_filename = models.generate_model_filename(dataset_name)
    print(f"Saving model to: {model_filename}")

    # Add image embeddings
    # images, labels = select_n_random(trainset.data, trainset.targets, 1000)
    # add_image_embeddings(writer, images, labels, classes)

    perf = PerformanceMonitor()
    test_perf = PerformanceMonitor()

    testloader_gen = iter(testloader)

    for epoch in range(params['num_epochs']):

        test_loss = 0
        accuracy = 0

        print()
        print(f"======== EPOCH {epoch}/{params['num_epochs']} ========")
        print()
        steps_per_epoch = len(trainloader)

        # Create a progress bar
        progress = tqdm(enumerate(trainloader, 0), total=steps_per_epoch, colour='green')

        # Iterate through steps in this Epoch
        for step, [input, targets] in progress:
            model.train()
            perf.reset()

            input, targets = input.to(device), targets.to(device)

            output = model(input)
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
            if ((global_step + 1) % params['decay_every_steps']) == 0:
                # decrease the learning rate
                scheduler.step()

            # Is it time to log?
            if (global_step % log_every) == 0:
                wandb_log({
                    "Loss/train": loss.item(),
                    "Accuracy/train": perf.get_accuracy()},
                    global_step,
                    epoch)

                wandb_log({"Learning Rate": scheduler.get_last_lr()[0]}, global_step, epoch)

            if ((global_step + 1) % params['test_every_steps']) == 0:
                # perform validation on test dataset
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

                wandb_log({
                    "Loss/test": test_perf.get_avg_loss(),
                    "Accuracy/test": test_accuracy},
                    global_step, epoch)


            avg_loss = perf.get_avg_loss()
            desc = f"Step: {step} / Loss: "
            desc += f"{avg_loss:.3f}" if avg_loss else "n/a"
            desc += f" / Accy: {perf.get_accuracy():.3f} "
            progress.set_description(desc)

            # end for each step in epoch

        # At the end of every epoch, we save the model
        models.save_model(model, model_filename)

    # end for each epoch

    print('Finished Training')

def wandb_log(params, step=None, epoch=None):
    if step is not None:
        params['batch'] = step
    if epoch is not None:
        params['epoch'] = epoch

    wandb.log(params)


def weights_init_uniform_rule(model):
    classname = model.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = model.in_features
        y = 1.0 / np.sqrt(n)
        model.weight.data.uniform_(-y, y)
        model.bias.data.fill_(0)


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
    num_classes = 254
    dataset_type = AsciiC64
    dataset_name = 'ascii_c64'

    # data_utils.write_dataset_class_counts(f'lib/datasets/{dataset_name}/data/{dataset_name}_class_counts', num_classes, dataset_type)
    train_model(num_classes, dataset_type, dataset_name)
