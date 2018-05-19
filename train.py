from __future__ import print_function, division


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

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
args = {
    "num_labels": 250
}

args = Struct(**args)

model = SqueezeNet(args)
model.type(torch.cuda.FloatTensor)

dataloaders = {'train': EitzDataLoader(16, 16, 'train'), 
               'val': EitzDataLoader(16, 16, 'val'), 
               'test': EitzDataLoader(16, 16, 'test')}

device = 'cuda'

dataset_sizes = {'train': 15000,
                'val': 2500,
                'test': 2500}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=.5)

best_model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

torch.save(model.state_dict(), f'/home/robincheong/sbir/checkpoints/SqueezeNet/eitz2012/{best_loss}')
