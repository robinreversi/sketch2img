from __future__ import print_function, division

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import SqueezeNet, ResNet

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import copy
from utils import get_dataloaders, get_default_parser, load_sketchy_images, get_loss_fn
from tensorboardX import SummaryWriter

def train_model(args):
    dataloaders = get_dataloaders(args)

    dataset_sizes = {'train': len(dataloaders['train'].dataset),
                'val': len(dataloaders['val'].dataset),
                'test': len(dataloaders['test'].dataset)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)

    model.to(device)
    
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
    
    criterion = get_loss_fn(args.dataset, args.loss_type)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloaders['train']) // 3, gamma=.9)
    
    writer = SummaryWriter(args.log_dir + "/{}".format(args.name))
    
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    batch_loss = float('inf')
    batch_num = 0
    
    now = datetime.datetime.now()
    torch.save(model.state_dict(), Path(args.save_dir) / f"{now.month}{now.day}{now.hour}{now.minute}{best_loss}")

    save_dir = Path(args.save_dir) / ('{}'.format(args.name))
    if not save_dir.exists():
        os.mkdir(save_dir)
    
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
            running_embedding_loss = 0.0
            running_classification_loss = 0.0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                if args.dataset == "eitz":
                    inputs = inputs.float()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        scheduler.step(loss)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                elif args.dataset == "sketchy":
                    # converts list of tuples of images paths of length N into flattened
                    # tensor of size N * args.loss_type 
                    N = len(inputs)
                    inputs = load_sketchy_images(inputs, args.loss_type, device, args.img_size)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        features = model.extract_features(inputs)
                        indices = torch.tensor(range(0, 2 * N)).to(device)
                        selected_features = torch.index_select(features, 0, indices)
                        logits = model.make_predictions(selected_features)
                        sketch_logits, photo_logits = torch.split(logits, N)
                        _, sketch_preds = torch.max(sketch_logits, 1)
                        _, photo_preds = torch.max(photo_logits, 1)
                        
                        sketch_cor = sum(sketch_preds.cpu().numpy() == labels.cpu().numpy()) 
                        photo_cor = sum(photo_preds.cpu().numpy() == labels.cpu().numpy())
                        
                        if args.loss_type == "classify":
                            loss = criterion(sketch_logits, photo_logits, labels)
                        else:
                            # reorganize into photo embeds and sketch embeds
                            # feed in embed for photo and sketch
                            embedding_loss, classification_loss = criterion(*torch.split(features, N), 
                                                                            sketch_logits, photo_logits, labels)
                            
                            loss = args.alpha * embedding_loss + (1 - args.alpha) * classification_loss 

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            
                    
                    
                    print("=" * 100)
                    print("Batch loss: {}".format(loss.item()))
                    print("Predicted classes for sketches: {}".format(sketch_preds.cpu().tolist()))
                    print("Predicted classes for photos: {}".format(photo_preds.cpu().tolist()))
                    print("Ground truth: {}".format(labels.cpu().tolist()))
                    print()
                    print("Sketch correct: {}".format(sketch_cor))
                    print("Photo correct: {}".format(photo_cor))
                    print("Overall correct: {}".format(photo_cor + sketch_cor))
                    print()
                    
                    if args.loss_type != "classify":
                        print("Classification loss: {}".format(classification_loss.item()))
                        print("Embedding Loss: {}".format(embedding_loss.item()))
                    
                    print("=" * 100)
                                        
                    if phase == "train":
                        writer.add_scalar('Batch/Total Loss', loss.item(), batch_num)
                        writer.add_scalar('Batch/Sketch Correct', sketch_cor, batch_num)
                        writer.add_scalar('Batch/Photo Correct', photo_cor, batch_num)
                        writer.add_scalar('Batch/Total Correct', sketch_cor + photo_cor, batch_num)
                        
                        if args.loss_type != "classify":
                            writer.add_scalar('Batch/Embedding Loss', embedding_loss.item(), batch_num)
                            writer.add_scalar('Batch/Classification Loss', classification_loss.item(), batch_num)
                        
                        batch_num += 1
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += photo_cor + sketch_cor
                    
                    if args.loss_type != "classify":
                        running_embedding_loss += embedding_loss  
                        running_classification_loss += classification_loss

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            epoch_acc = running_corrects / (2 * dataset_sizes[phase])
            print('{} Acc: {:.4f}'.format(phase, epoch_acc))
            
            writer.add_scalar('{}/Total Loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('{}/Classification Loss'.format(phase), running_classification_loss, epoch)
            writer.add_scalar('{}/Acc'.format(phase), epoch_acc, epoch)
            
            if args.loss_type != "classify":
                print('{} Embed Loss: {:.4f}'.format(phase, running_embedding_loss))
                writer.add_scalar('{}/Embedding Loss'.format(phase), running_embedding_loss, epoch)

            
            # deep copy the model
            if phase == 'val':
                if args.loss_type != 'classify':
                    if running_embedding_loss < best_loss:
                        best_loss = running_embedding_loss
                        now = datetime.datetime.now()
                        torch.save(model.state_dict(), save_dir / f"{now.month}{now.day}{now.hour}{now.minute}_{best_loss}")
                        best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        now = datetime.datetime.now()
                        torch.save(model.state_dict(), save_dir / f"{now.month}{now.day}{now.hour}{now.minute}_{best_loss}")
                        best_model_wts = copy.deepcopy(model.state_dict())     
        
        epoch_time_lapse = time.time() - epoch_start
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_lapse // 60, epoch_time_lapse % 60))
        print()

    writer.close()
    now = datetime.datetime.now()
    torch.save(model.state_dict(), save_dir / f"end_{now.month}{now.day}{now.hour}{now.minute}_{best_loss}")
        
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    now = datetime.datetime.now()
    
    torch.save(model.state_dict(), save_dir / "best")

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--alpha', required=True, type=float, help='weighting for embedding vs classification loss')
    parser.add_argument('--lr', default=1e-2, type=float, help="learning rate to start with")
    parser.add_argument('--wd', default=5e-3, type=float, help="l2 reg term")
    parser.add_argument('--loss_type', type=str, required=True, choices=('classify', 'binary', 'trip', 'quad'), 
                        help='which type of contrastive loss to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
    parser.add_argument('--toy', action='store_true', help='Use reduced dataset if true.')
    parser.add_argument('--toy_size', type=int, default=5,
                        help='How many of each type to include in the toy dataset.')
    parser.add_argument('--save_dir', type=str, default='/home/robincheong/sketch2img/ckpts/',
                        help='Directory in which to save checkpoints.')
    parser.add_argument('--log_dir', type=str, required=True, default="logs/",
                        help="directory to save the tensorboard log files to")
    parser.add_argument('--name', type=str, required=True, help='name to use for tensorboard logging')


    args = parser.parse_args()
    train_model(args)
    
