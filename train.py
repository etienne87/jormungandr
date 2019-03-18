from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
from models import resnet
import matplotlib.pyplot as plt
import time
import os
import copy
from textwrap import wrap
import re
import itertools
import dataset
import utils
from sklearn.metrics import f1_score, confusion_matrix
from tensorboardX import SummaryWriter

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def draw_cfm(cm, writer, labels, phase, global_step):
    fig = plt.Figure(figsize=(7, 7), dpi=160, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', str(x)) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    writer.add_figure(phase + '_confusion_matrix', fig, global_step=global_step)


def draw_stn(model, batch, writer):
    batch2 = model.show_stn(batch)

    #denormalize
    mean = [0.485, 0.456, 0.406][::-1]
    std = [0.229, 0.224, 0.225][::-1]
    batch2 = batch2 * std + mean




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, progressive_resize=False):
    since = time.time()

    val_acc_history = []
    val_f1_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Progressive resize
        if progressive_resize:
            factor = max(0.1, epoch/num_epochs)
            imsize = (int(600 * factor), int(600 * factor))
            print('current imsize: ', imsize)
            dataloaders['train'].dataset.transform[0].imsize = imsize

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data.
            for iter, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                corrects = torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrects

                #for sklearn global statistics
                y_true += labels.data.cpu().numpy().tolist()
                y_pred += preds.data.cpu().numpy().tolist()

                if iter%100 == 0:
                    print('{} Loss: {:.4f} Acc: {:.4f} {:d} / {:d}'.format(phase,
                                                                loss.item(), float(corrects) / preds.shape[0],
                                                                iter, len(dataloaders[phase]) ) )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            labels = np.arange(y_true.max())
            epoch_f1 = f1_score(y_true, y_pred, labels=labels, average='micro')
            epoch_cfm = confusion_matrix(y_true, y_pred, labels=labels)

            writer.add_scalar(phase + '_loss', epoch_loss, epoch)
            writer.add_scalar(phase + '_acc', epoch_acc, epoch)
            writer.add_scalar(phase + '_f1', epoch_f1, epoch)
            draw_cfm(epoch_cfm, writer, labels, phase=phase, global_step=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(phase, ' F1: ', epoch_f1)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_f1_history.append(epoch_f1)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




if __name__ == '__main__':
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "/home/etienneperot/workspace/datasets/snakes/"
    model_name = "resnet"
    num_classes = 45
    batch_size = 32
    num_epochs = 15
    feature_extract = True

    model_ft = resnet(num_classes=num_classes, pretrained=True, resnet_model='resnet34', add_stn=True)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.485, 0.456, 0.406][::-1]
    std = [0.229, 0.224, 0.225][::-1]
    input_size = (600, 600)
    data_transforms = {'train':
            dataset.Compose([
            dataset.ResizeCV(input_size),
            dataset.RandomDA(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ]),
    'val':
        dataset.Compose([
        dataset.ResizeCV(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])}

    print("Initializing Datasets and Dataloaders...")
    root = '/home/etienneperot/workspace/datasets/snakes/train/'
    file_paths = {'train': os.path.join(root, "train.pkl"),
                  'val': os.path.join(root, "val.pkl")
                  }

    print(file_paths['train'])

    # Create training and validation datasets
    image_datasets = {x: dataset.SnakeDataset(file_paths[x], data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('device: ', device)
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    # Save it in a folder with the name of the experience
