from datetime import datetime
from tqdm import tqdm, trange

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from mnist_network import ClassifierNetwork
from performance_monitor import PerformanceMonitor
from tensorboard_writer import TensorboardWriter

models_path = './models/'

def do_main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA driver")
    else:
        device = torch.device('cpu')
        print("No CUDA driver available. Using CPU fallback")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    batch_size = 64

    """ Load the train and test datasets """

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0,  pin_memory=True)

    # testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=0)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    train(trainset, trainloader, device, classes)


def train(trainset, trainloader, device, classes):
    net = ClassifierNetwork()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer = TensorboardWriter("logs/", trainset)

    # Add image embeddings
    # images, labels = select_n_random(trainset.data, trainset.targets, 1000)
    # add_image_embeddings(writer, images, labels, classes)

    global_step = 0
    perf = PerformanceMonitor()

    for epoch in range(1):
        print()
        print(f"=== EPOCH {epoch} ========")
        print()

        # Create a progress bar
        progress = tqdm(enumerate(trainloader, 0), total=len(trainloader), colour='green')

        for step, data in progress:
            perf.reset()

            input, labels = data
            input, labels = input.to(device), labels.to(device)

            # Show NN graph in Tensorboard
            # writer.add_graph(net, inputs)

            output = net(input)
            loss = criterion(output, labels)
            perf.loss.append(loss.item())

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training stats
            perf.add_predictions(output, labels)

            global_step = epoch * len(trainloader) + step
            writer.add_scalar("Loss/train", loss.item(), global_step)

            # Log accuracy
            writer.add_scalar("Accuracy/train", perf.get_accuracy(), global_step)

            desc = f"Step: {step} / Loss: "

            avg_loss = perf.get_avg_loss()
            desc += f"{avg_loss:.3f}" if avg_loss else "n/a"
            desc += f" / Accy: {perf.get_accuracy():.3f} "

            progress.set_description(desc)


    writer.flush()
    writer.close()
    print('Finished Training')

    model_file = models_path + 'mnist_net.pth'
    # torch.save(net.state_dict(), model_file)
    exit(1)

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
    # print("Exiting")
    do_main()
