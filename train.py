from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as tvutils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy
from pathlib import Path

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from collections import defaultdict

from models.models import SqueezeNet, ResNet, ConvDualVAE, ConvDualAE, ConvSingleAE, ConvSingleVAE
from utils import get_dataloaders, get_default_parser, \
                    load_sketchy_images, log_metrics, get_train_parser

from model_utils import get_loss_fn, load_model, vae_forward, ae_forward, \
                        classify_contrast_forward, gan_forward
    
from loaders.datasets.constants import *
        
def train_model(args):
    dataloaders = get_dataloaders(args)

    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                'val': len(dataloaders['val'].dataset),
                'test': len(dataloaders['test'].dataset)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set up
    model = load_model(args, device)
    loss_fn = get_loss_fn(args.dataset, args.loss_type)
    
    if args.train_decoders:
        parameters = list(model.photo_decoder.parameters()) + list(model.sketch_decoder.parameters())
    elif args.model in ['EmbedGAN']:
        parameters = list(model.G.parameters()) + list(model.D.parameters())
    else:
        parameters = model.parameters()
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=.9, nesterov=True)
    elif args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloaders['train']) // 10, gamma=.9)
    writer = SummaryWriter(args.log_dir + "/{}".format(args.name))


    save_dir = Path(args.save_dir) / ('{}'.format(args.name))
    if not save_dir.exists():
        os.mkdir(save_dir)
    
    best_model = None
    best_loss = float('inf')
    batch_num = 0
    
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            epoch_metrics = defaultdict(float)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()

                N = len(inputs)
                
                # converts list of tuples of images paths of length N into flattened
                # tensor of size N * args.loss_type 
                inputs = load_sketchy_images(inputs, args.loss_type, device, args.img_size)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    if args.loss_type in ["vae", "vae+embed", "vae+embed+classify"]:
                        batch_metrics = vae_forward(inputs, labels, model, loss_fn, writer, device, 
                                                    batch_num, args.alpha, N, args.name, modality=args.modality,
                                                    compare_embed=args.loss_type in ["vae+embed", "vae+embed+classify"],
                                                    classify=args.loss_type in ['vae+embed+classify', 'single_vae'])
                    elif args.loss_type in ["ae", "ae+embed", "ae+embed+classify"]:
                        batch_metrics = ae_forward(inputs, labels, model, loss_fn, writer, device, 
                                                   batch_num, args.alpha, N, args.name, modality=args.modality,
                                                   compare_embed=args.loss_type in ["ae+embed", "ae+embed+classify"],
                                                   classify=args.loss_type in ['ae+embed+classify', 'single_ae'])
                    elif args.loss_type in ['gan']:
                        batch_metrics = gan_forward(inputs, labels, model, loss_fn, writer, device, batch_num, N)
                    else:
                        batch_metrics = classify_contrast_forward(inputs, labels, model, loss_fn, writer, 
                                                                  device, batch_num, args.alpha, args.loss_type, N)
                        
                    for criteria_name in batch_metrics:
                        epoch_metrics[criteria_name] += batch_metrics[criteria_name] / dataset_sizes[phase]
                    
                    loss = batch_metrics['loss']

                    del batch_metrics

                    if phase == "train":
                        batch_num += 1
                        loss.backward()
                        optimizer.step()
            
            epoch_loss = epoch_metrics['loss'].item()
            log_metrics(epoch_metrics, writer, phase, epoch)

        
        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            now = datetime.datetime.now()
            torch.save(model.state_dict(), save_dir / f"{now.month}{now.day}{now.hour}{now.minute}_{best_loss}")
            best_model = copy.deepcopy(model.state_dict())             

            
    writer.close()
    now = datetime.datetime.now()
    torch.save(model.state_dict(), save_dir / f"end_{now.month}{now.day}{now.hour}{now.minute}_{best_loss}")

    # load best model weights
    model.load_state_dict(best_model)
    now = datetime.datetime.now()
    torch.save(model.state_dict(), save_dir / "best")

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    train_model(args)
    
