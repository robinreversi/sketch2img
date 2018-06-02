from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

from models.models import SqueezeNet, ResNet, ConvVAE
from utils import get_dataloaders, get_default_parser, load_sketchy_images, get_loss_fn
from loaders.datasets.constants import *


def load_model(args, device):
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)
    elif args.model == "ConvVAE":
        ftr_extractor = None
        if args.ftr_extractor_path:
            ftr_extractor = ResNet()
            ftr_extractor.load_state_dict(torch.load(args.ftr_extractor_path))
        model = ConvVAE(args, ftr_extractor)

    model.to(device)
    
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
        
    return model


def log_metrics(metrics, writer, stage, idx):
    """ Logs the metrics stored in metrics. 
    
    Args: 
        metrics: a dict containing metric name -> metric val (tensor)
        writer: a tensorboard writer
        stage: one of 'batch', 'train', 'val'
        idx: batch_num or epoch
    """
    
    print('=' * 50)
    for metric_name in metrics:
        metric_val = metrics[metric_name]
        if type(metric_val) == torch.Tensor:
            metric_val = metric_val.item()
        print('{}_{}: {}'.format(stage, metric_name, metric_val))
        writer.add_scalar('{}/{}'.format(stage, metric_name), metric_val, idx)
            
            
def vae_forward(inputs, labels, model, loss_fn, writer, device, batch_num, alpha, N):
    metrics = {}
    sketches = torch.index_select(inputs, 0, torch.tensor(range(0, N)).to(device))
    photos = torch.index_select(inputs, 0, torch.tensor(range(N, 2 * N)).to(device))
    recon_sketch, sketch_mu, sketch_logvar = model.forward(sketches, is_sketch=True)
    recon_photo, photo_mu, photo_logvar = model.forward(photos, is_sketch=False)

    metrics['sketch_kl_divergence'], metrics['sketch_recon_loss'] = loss_fn(recon_sketch, sketches,
                                                                              sketch_mu, sketch_logvar)

    metrics['photo_kl_divergence'], metrics['photo_recon_loss'] = loss_fn(recon_photo, photos, 
                                                                            photo_mu, photo_logvar)

    metrics['sketch_loss'] = metrics['sketch_kl_divergence'] + metrics['sketch_recon_loss']

    metrics['photo_loss'] = metrics['photo_kl_divergence'] + metrics['photo_recon_loss']

    metrics['loss'] = alpha * metrics['sketch_loss'] + (1-alpha) * metrics['photo_loss']
    
    log_metrics(metrics, writer, "batch", batch_num)
    
    return metrics


def classify_contrast_forward(inputs, labels, model, loss_fn, writer, device, 
                              batch_num, alpha, loss_type, N):
    metrics = {}
    features = model.extract_features(inputs)
    indices = torch.tensor(range(0, 2 * N)).to(device)
    selected_features = torch.index_select(features, 0, indices)
    logits = model.make_predictions(selected_features)
    sketch_logits, photo_logits = torch.split(logits, N)

    if loss_type == "classify":
        metrics['loss'] = loss_fn(sketch_logits, photo_logits, labels)
    else:
        # reorganize into photo embeds and sketch embeds
        # feed in embed for photo and sketch
        metrics['embedding_loss'], metrics['classification_loss'] = loss_fn(*torch.split(features, N), 
                                                                                sketch_logits, photo_logits, labels)

        metrics['loss'] = alpha * metrics['embedding_loss'] + (1 - alpha) * metrics['classification_loss']

    _, sketch_preds = torch.max(sketch_logits, 1)
    _, photo_preds = torch.max(photo_logits, 1)

    sketch_cor = sum(sketch_preds.cpu().numpy() == labels.cpu().numpy()) 
    photo_cor = sum(photo_preds.cpu().numpy() == labels.cpu().numpy())

    metrics['sketch_cor'] = sketch_cor
    metrics['photo_cor'] = photo_cor

    log_metrics(metrics, writer, "batch", batch_num)

    # TODO Change this to be args.verbose
    if True:
        print("=" * 100)
        print("Predicted classes for sketches: {}".format(sketch_preds.cpu().tolist()))
        print("Predicted classes for photos: {}".format(photo_preds.cpu().tolist()))
        print("Ground truth: {}".format(labels.cpu().tolist()))
        print("=" * 100)
                    

    return metrics


def get_criteria(loss_type):
    if loss_type == "vae":
        return VAE_CRITERIA
    elif loss_type == "classify":
        return CLASSIFY_CRITERIA
    elif loss_type in ['binary', 'trip', 'quad']:
        return CLASSIFY_CONTRAST_CRITERIA
    else:
        raise ValueError

        
def train_model(args):
    dataloaders = get_dataloaders(args)

    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                'val': len(dataloaders['val'].dataset),
                'test': len(dataloaders['test'].dataset)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set up
    model = load_model(args, device)
    loss_fn = get_loss_fn(args.dataset, args.loss_type)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloaders['train']) // 3, gamma=.9)
    writer = SummaryWriter(args.log_dir + "/{}".format(args.name))
    criteria = get_criteria(args.loss_type)

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
                    if args.loss_type == "vae":
                        batch_metrics = vae_forward(inputs, labels, model, loss_fn, writer, device, 
                                                    batch_num, args.alpha, N)
                    else:
                        batch_metrics = classify_contrast_forward(inputs, labels, model, loss_fn, writer, 
                                                                  device, batch_num, args.alpha, args.loss_type, N)
                        
                    for criteria_name in criteria:
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
    parser = get_default_parser()
    parser.add_argument('--alpha', required=True, type=float, help='changes various loss weightings depending on model type')
    parser.add_argument('--lr', default=1e-2, type=float, help="learning rate to start with")
    parser.add_argument('--wd', default=5e-3, type=float, help="l2 reg term")
    parser.add_argument('--loss_type', type=str, required=True, choices=('classify', 'binary', 'trip', 'quad', 'vae'), 
                        help='which type of contrastive loss to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
    parser.add_argument('--toy', action='store_true', help='Use reduced dataset if true.')
    parser.add_argument('--toy_size', type=int, default=5,
                        help='how many of each type to include in the toy dataset.')
    parser.add_argument('--save_dir', type=str, default='/home/robincheong/sketch2img/ckpts/',
                        help='directory in which to save checkpoints.')
    parser.add_argument('--log_dir', type=str, required=True, default="logs/",
                        help="directory to save the tensorboard log files to")
    parser.add_argument('--name', type=str, required=True, help='name to use for tensorboard logging')
    parser.add_argument('--h_size', type=int, default=512, help="number of hidden units to use in the VAE")
    parser.add_argument('--ftr_extractor_path', type=str, default='', 
                        help='path to model to use as a feature extractor in the VAE')
    args = parser.parse_args()
    train_model(args)
    
