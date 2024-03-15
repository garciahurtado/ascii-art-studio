import os

import numpy as np
import cv2 as cv
import torch
import torch.utils
import torch.utils.data
import torchmetrics
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time

from datasets.ascii_c64 import AsciiC64
import datasets.data_utils as data
import pytorch.model_manager as models

def eval_model(dataset_type, dataset_name, model_filename):
    num_classes = 308
    batch_size = 4096
    device = data.get_device()

    model = models.load_model(dataset_name, model_filename, num_labels=num_classes)
    model.to(device)
    model.eval()  # put the model in inference mode

    # Load the test dataset
    dataset = data.get_dataset(train=False, dataset_type=dataset_type, num_labels=num_classes)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False)

    # Initialize metrics
    metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='weighted')
    metric_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='weighted')
    metric_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='weighted')
    metric_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted')

    metric.to(device)
    metric_precision.to(device)
    metric_recall.to(device)
    metric_f1.to(device)

    total_steps = len(dataloader)
    total_samples = total_steps * batch_size

    with torch.no_grad():
        start = time.time()
        print(f"Starting inference of {total_samples} items...")

        # Create a progress bar
        progress = tqdm(enumerate(dataloader, 0), total=total_steps, colour='green')
        all_output_labels = []
        all_true_targets = []

        for step, [input, target] in progress:
            input, target = input.to(device), target.to(device)
            true_targets = torch.argmax(target, dim=1)
            all_true_targets.extend(true_targets.cpu().numpy())

            outputs = model(input)
            output_labels = torch.argmax(outputs, dim=-1)  # Convert from one-hot to indexes
            all_output_labels.extend(output_labels.cpu().numpy())

            metric.update(output_labels, true_targets)
            metric_precision.update(output_labels, true_targets)
            metric_recall.update(output_labels, true_targets)
            metric_f1.update(output_labels, true_targets)

    # Calculate accuracy per class
    report = classification_report(all_true_targets, all_output_labels, output_dict=True)

    data_root = dataset.data_root
    data_root = os.path.dirname(data_root)
    filename = os.path.join(data_root, f'{model_filename}-class-accuracy.txt')
    print(f"Saving class accuracy to {filename}....")

    with open(filename, "w") as file:
        for class_name, class_report in report.items():
            if class_name != 'accuracy':
                precision = f"{class_report['precision']*100:.2f}"
                file.write(f"{class_name},{precision}\n")
                print(f"Class {class_name}: Accuracy = {precision}")

    accuracy = metric.compute() * 100
    end = time.time()
    duration = end - start
    time_per_label = duration / total_samples
    preds_per_second = float(1 / time_per_label)

    print()
    print("=== RESULTS ===")
    print(f"Correct xxx/{total_samples}: {accuracy:.3f}% accuracy.")
    print(f"Precision: {metric_precision.compute() * 100}")
    print(f"Recall: {metric_recall.compute() * 100}")
    print(f"F1 Score: {metric_f1.compute() * 100}")
    print()
    print(f"Inferred {total_samples} in {duration:.2f} seconds ({preds_per_second:.2f}/s)")
    print()

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
    min, _ = torch.min(kernels, 1)
    max, _ = torch.max(kernels, 1)
    kernels = kernels - min
    kernels = kernels / max
    filter_img = torchvision.utils.make_grid(kernels, nrow=12)
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
    dataset_type = AsciiC64
    dataset_name = 'ascii_c64'
    model_filename = 'ascii_c64-Mar10_03-56-51'

    eval_model(dataset_type, dataset_name, model_filename)