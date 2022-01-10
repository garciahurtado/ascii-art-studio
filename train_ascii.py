import numpy as np
import torchmetrics
import os
import sys
from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import wandb

sys.path.append('./lib')

from torchvision.transforms import Lambda
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets.ascii_amstrad_cpc.ascii_amstrad_cpc import AsciiAmstradCPC
from net.ascii_classifier_network import AsciiClassifierNetwork
from performance_monitor import PerformanceMonitor
from tensorboard_writer import TensorboardWriter
import datasets.data_utils as data_utils

models_path = os.path.normpath('./models/')
num_labels = 511

def do_main():
    # Eliminate randomness to increase training reproducibility
    torch.manual_seed(777)
    np.random.seed(777)

    params = {
        'batch_size':       1024,
        'num_epochs':       50,
        'learning_rate':    0.002,
        'decay_rate':       0.96,
        'decay_every_steps':2000,
        'test_every_steps': 30
    }

    transform = transforms.Compose(
        [transforms.ToTensor()])
    target_transform = transforms.Compose(
        [data_utils.OneHot()])

    """ Load the train and test datasets """
    dataset = AsciiAmstradCPC(
        transform=transform,
        target_transform=target_transform,
        train=True,
        device=get_device())

    trainset, testset = split_dataset(dataset)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True)

    train(dataset, trainloader, testloader, params)

def split_dataset(dataset, test_split=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    trainset = Subset(dataset, train_idx)
    testset = Subset(dataset, test_idx)
    return trainset, testset

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA driver")
    else:
        device = torch.device('cpu')
        print("No CUDA driver available. Using CPU fallback")

    return device

def eval_model():
    model_name = 'AsciiAmstradCPC-Jan06_23-57-44.pth'
    device = get_device()

    transform = transforms.Compose(
        [transforms.ToTensor()])
    target_transform = transforms.Compose(
        [data_utils.one_hot])

    # Load the test dataset
    dataset = AsciiAmstradCPC(transform=transform, target_transform=target_transform, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=4,
        pin_memory=False)

    model_file = os.path.join(models_path, 'AsciiAmstradCPC', model_name)
    model = torch.load(model_file)
    model.to(device)
    model.eval()

    metric = torchmetrics.Accuracy(num_classes=512)
    metric.to(device)

    with torch.no_grad():
        input, labels = next(iter(dataloader))
        start = time.time()

        input, labels = input.to(device), labels.to(device)
        total = input.size(0)
        labels = torch.argmax(labels, dim=1)

        print(f"Starting inference of {total} items...")
        results = model(input)

        accy = metric(results, labels)* 100

        end = time.time()
        duration = end - start
        time_per_label = duration / metric.total
        preds_per_second = float(1 / time_per_label)

        print(f"Correct {metric.correct}/{metric.total}: {accy:.3f}% accuracy.")
        print(f"Inferred {metric.total} in {duration:.2f} seconds ({preds_per_second:.2f}/s)")

def train(trainset, trainloader, testloader, params):
    global models_path

    device = get_device()

    model = AsciiClassifierNetwork()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Logging
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key='e6533f84f309fe9d42fd1e3577f8ba162ed921c0')
    wandb.init(
        project='pytorch-amstrad-cpc',
        entity='ghurtado',
        config=params)

    # writer = TensorboardWriter("logs/", trainset)
    # writer.add_hparams(params, {})
    dataset_name = type(trainset).__name__
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

    model_file = os.path.join(models_path, dataset_name, f"{dataset_name}-" + timestamp + '.pth')
    print(f"Saving model file to: {model_file}")

    # Add image embeddings
    # images, labels = select_n_random(trainset.data, trainset.targets, 1000)
    # add_image_embeddings(writer, images, labels, classes)

    perf = PerformanceMonitor()
    test_perf = PerformanceMonitor()
    log_every = 10 # Log every N steps
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params['decay_rate'])
    testloader = iter(testloader)

    for epoch in range(params['num_epochs']):
        print()
        print(f"======== EPOCH {epoch} ========")
        print()
        steps_per_epoch = len(trainloader)
        global_step = 0

        # Create a progress bar
        progress = tqdm(enumerate(trainloader, 0), total=steps_per_epoch, colour='green')

        for step, [input, labels] in progress:
            model.train()
            perf.reset()
            input, labels = input.to(device), labels.to(device)

            # Show NN graph in Tensorboard
            # writer.add_graph(model, inputs)

            output = model(input)
            loss = criterion(output, labels)
            perf.loss.append(loss.item())

            # zero out the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training stats
            labels = torch.argmax(labels, dim=1, keepdim=True)
            perf.add_predictions(output, labels)

            global_step = (epoch * steps_per_epoch) + step

            if (global_step % params['decay_every_steps']) == 0:
                # decrease the learning rate
                scheduler.step()

            if (global_step % log_every) == 0:
                wandb_log({
                    "Loss/train": loss.item(),
                    "Accuracy/train": perf.get_accuracy()},
                    global_step,
                    epoch)

                wandb_log({"Learning Rate": scheduler.get_last_lr()[0]}, global_step, epoch)

            if ((global_step+1) % params['test_every_steps']) == 0:
                # perform validation on test dataset
                model.eval()
                test_perf.reset()
                [input, labels] = next(testloader)
                input, labels = input.to(device), labels.to(device)

                output = model(input)
                test_loss = criterion(output, labels)
                test_perf.loss.append(test_loss.item())

                # Calculate the average test loss
                labels = torch.argmax(labels, dim=1, keepdim=True)
                test_perf.add_predictions(output, labels)

                wandb_log({
                    "Loss/test" : test_loss.item(),
                    "Accuracy/test": test_perf.get_accuracy()},
                    global_step, epoch)

            desc = f"Step: {step} / Loss: "

            avg_loss = perf.get_avg_loss()
            desc += f"{avg_loss:.3f}" if avg_loss else "n/a"
            desc += f" / Accy: {perf.get_accuracy():.3f} "

            progress.set_description(desc)

        # At the end of every epoch, we save the model
        torch.save(model, model_file)

    print('Finished Training')

def wandb_log(params, step=None, epoch=None):
    if(step is not None):
        params['batch'] = step
    if(epoch is not None):
        params['epoch'] = epoch

    wandb.log(params)

def validate(model, testloader, criterion, labels):
    test_loss = 0
    accuracy = 0

    for inputs, classes in testloader:
        inputs = inputs.to('cuda')
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

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
    max_scores, max_idx_class = model(x).max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    # usually 0th coordinate is batch size
    n = x.size(0)
    assert( n == max_idx_class.size(0))
    # calulate acc (note .item() to do float division)
    acc = (max_idx_class == y).sum().item() / n
    return acc

if __name__ == "__main__":
    do_main()
    # eval_model()
