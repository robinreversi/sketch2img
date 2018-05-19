from __future__ import print_function, division

import datetime
from loaders.EitzDataLoader import EitzDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import SqueezeNet

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import get_dataloaders, get_default_parser

def train_model(args):
    dataloaders = get_dataloaders(args)

    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                'val': len(dataloaders['val'].dataset),
                'test': len(dataloaders['test'].dataset)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=.5)
    
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        
        epoch_start = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        epoch_time_lapse = time.time() - epoch_starts
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_lapse // 60, epoch_time_lapse % 60))
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    now = datetime.datetime.now()
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"{now.month}{now.day}{now.hour}{now.minute}"))

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    train_model(args)
    