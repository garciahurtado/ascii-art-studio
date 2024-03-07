import numpy as np
import pandas as pd
import torchvision
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
import torchmetrics
import os
import sys
from datetime import datetime
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import wandb
from focal_loss import FocalLoss

sys.path.append('./lib')

from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

from datasets.ascii_amstrad_cpc.ascii_amstrad_cpc import AsciiAmstradCPC
from net.ascii_classifier_network import AsciiClassifierNetwork
from performance_monitor import PerformanceMonitor
import datasets.data_utils as data_utils
import pytorch.model_manager as models

def do_main():
    # Eliminate randomness to increase training reproducibility
    torch.manual_seed(123456)
    np.random.seed(123456)
    batch_size = 512
    test_batch_size = 2048

    params = {
        'batch_size':       batch_size,
        'num_epochs':       20,
        'learning_rate':    0.0001,
        'decay_rate':       0.96, # todo: calculate from batch size and total number of samples
        'decay_every_steps': 512,
        'test_every_steps': 128
    }

    random_state = 66

    trainset = get_dataset(train=True)
    testset = get_dataset(train=False)

    #trainset, testset = split_dataset(trainset, 0.2, random_state=random_state)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2,
        prefetch_factor=8,
        drop_last=True)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=8,
        drop_last=True,
        worker_init_fn=data_utils.seed_init_fn)

    train(trainset.get_class_counts(), trainloader, testloader, params)

def split_dataset(dataset, test_split=0.2, random_state=0):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=random_state)
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

def train(class_counts, trainloader, testloader, params):
    global models_path

    device = get_device()

    # Get class counts
    class_cardinality = len(class_counts)
    print(f"Found {class_cardinality} classes in dataset")

    # AsciiClassifierNetwork.calculate_padding(8, 8, 3, 3, 2, 2)
    model = AsciiClassifierNetwork(num_labels=class_cardinality)
    model.to(device)

    # Initialize parameter weights:
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    model.apply(weights_init_uniform_rule)

    # Generate class weights
    class_weights = data_utils.create_class_weights(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights.to(device)

    # Loss function / optimizer / Learning rate scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], fused=True, amsgrad=True, eps=1e-9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, params['decay_rate'])

    # Logging
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key='e6533f84f309fe9d42fd1e3577f8ba162ed921c0')
    wandb.init(
        project='pytorch-amstrad-cpc',
        entity='ghurtado',
        config=params)

    dataset_name = 'AsciiAmstradCPC'
    model_file = models.generate_model_filename(dataset_name)
    print(f"Saving model file to: {model_file}")

    # Add image embeddings
    # images, labels = select_n_random(trainset.data, trainset.targets, 1000)
    # add_image_embeddings(writer, images, labels, classes)

    perf = PerformanceMonitor()
    test_perf = PerformanceMonitor()
    log_every = 3 # Log every N steps


    testloader_gen = iter(testloader)

    for epoch in range(params['num_epochs']):

        print()
        print(f"======== EPOCH {epoch} ========")
        print()
        steps_per_epoch = len(trainloader)

        # Create a progress bar
        progress = tqdm(enumerate(trainloader, 0), total=steps_per_epoch, colour='green')

        for step, [input, labels] in progress:
            model.train()
            perf.reset()
            input, labels = input.to(device), labels.to(device)

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

            # +1 below makes sure we don't decay on step 0
            if ((global_step+1) % params['decay_every_steps']) == 0:
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

                try:
                    input, labels = next(testloader_gen)
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    testloader_gen = iter(testloader)
                    input, labels = next(testloader_gen)

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

def eval_model():
    dataset_name = 'AsciiAmstradCPC'
    model_name = 'AsciiAmstradCPC-Mar06_01-26-52.pt'
    num_classes = 486
    batch_size = 4096
    device = get_device()

    model = models.load_model(dataset_name, model_name )
    model.to(device)
    model.eval() # put the model in inference mode

    # Load the test dataset
    dataset = get_dataset(train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False)

    metric = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
    metric.to(device)
    total_steps = len(dataloader)

    with torch.no_grad():
        start = time.time()
        print(f"Starting inference of {total_steps} items...")

        # Create a progress bar
        progress = tqdm(enumerate(dataloader, 0), total=total_steps, colour='green')

        for step, [input, target] in progress:
            input, target = input.to(device), target.to(device)
            outputs = model(input)
            output_labels = torch.argmax(outputs, dim=-1)
            metric.update(output_labels, torch.argmax(target, dim=1))

        accuracy = metric.compute() * 100

        end = time.time()
        duration = end - start
        time_per_label = duration / total_steps
        preds_per_second = float(1 / time_per_label)

        print(f"Correct xxx/{total_steps}: {accuracy:.3f}% accuracy.")
        # print(f"Inferred {total_steps} in {duration:.2f} seconds ({preds_per_second:.2f}/s)")


def wandb_log(params, step=None, epoch=None):
    if(step is not None):
        params['batch'] = step
    if(epoch is not None):
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


def get_dataset(train=True):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    target_transform = transforms.Compose(
        [data_utils.OneHot()])

    dataset = AsciiAmstradCPC(
        transform=transform,
        target_transform=target_transform,
        train=train,
        device=get_device())

    return dataset

def write_dataset_class_counts():
    # Extract class counts for class weights

    print("Evaluating class counts... This may take some time")
    counts = data_utils.get_class_counts(get_dataset())
    pd.DataFrame(counts).to_csv("lib/datasets/ascii_amstrad_cpc/data/amstrad-cpc_class_counts.csv", header=False)

    counts = data_utils.get_class_counts(get_dataset(train=False))
    pd.DataFrame(counts).to_csv("lib/datasets/ascii_amstrad_cpc/data/amstrad-cpc_class_counts-test.csv", header=False)

def visualize_filters():
    # instantiate model
    conv = models.load_model('AsciiAmstradCPC', 'AmstradCPC-Mar03_22-25-56.pth')

    # load weights if they haven't been loaded
    # skip if you're directly importing a pretrained network
    # checkpoint = torch.load('model_weights.pt')
    # conv.load_state_dict(checkpoint)


    # get the kernels from the first layer
    # as per the name of the layer
    kernels = conv.conv1.weight.detach().clone()
    kernels.to("cpu")

    visTensor(kernels, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()

    return

    # normalize to (0,1) range so that matplotlib
    # can plot them
    min, _ = torch.min(kernels,1)
    max, _ = torch.max(kernels,1)
    kernels = kernels - min
    kernels = kernels / max
    filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
    # change ordering since matplotlib requires images to
    # be (H, W, C)
    img = filter_img.permute(1, 2, 0)
    plt.imshow(img)

    # You can directly save the image as well using
    # img = save_image(kernels, 'encoder_conv1_filters.png' ,nrow = 12)

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, h, w = tensor.shape
    print(f"Filter tensor shape: n:{n}, c:{c}, h:{h}, w:{w}")
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    mygrid = grid.to("cpu").numpy()
    plt.imshow(mygrid.transpose((1, 2, 0)))

if __name__ == "__main__":

    #write_dataset_class_counts()
    # do_main()
    # visualize_filters()
    eval_model()


