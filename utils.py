from loaders.EitzDataLoader import EitzDataLoader
from loaders.SketchyDataLoader import SketchyDataLoader
from loaders.datasets.constants import *
from PIL import Image
import random
import numpy as np
import torch.nn as nn
import argparse
import torch
import torchvision.transforms as T
import torchvision.utils as tvutils
from models.models import SqueezeNet, ResNet, ConvDualVAE, ConvDualAE, ConvSingleAE, ConvSingleVAE

def get_dataloaders(args):
    """ Returns a dict with dataloaders for each phase.
    
    Args:
        dataset: the dataset to use; one of 'eitz', 'sketchy'
    """
    if args.dataset == 'eitz':
        dataloaders = {'train': EitzDataLoader(args, 'train'), 
                       'val': EitzDataLoader(args, 'val'), 
                       'test': EitzDataLoader(args, 'test')}
    elif args.dataset == 'sketchy':
        dataloaders = {'train': SketchyDataLoader(args, 'train'), 
                       'val': SketchyDataLoader(args, 'val'), 
                       'test': SketchyDataLoader(args, 'test')}
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    return dataloaders


def load_sketchy_images(inputs, loss_type, device, img_size):
    """ Converts a list of file path tuples into corresponding tensor-images.
    
    Args:
        inputs: a list of file path tuples
        loss_type: the type of loss to use
        device: cuda or cpu
        img_size: rescale size
        
    Returns:
        torch.cat(all_images): a tensor organized such that the first N tensors
                                correspond to sketches, the next N tensors
                                correspond to gt images, the next N tensors
                                correspond to same cat diff photo images (if trip or quad loss), 
                                the next N tensors correspond to diff cat images (if quad loss)
    """
    images = [[], [], [], []]
    for example in inputs:
        example = example.split("++")
        # only compute on relevant images by removing unnecessary paths
        if loss_type in ["classify", "binary", 'vae', 'vae+embed', 
                         'vae+embed+classify', 'ae', 'ae+embed', 
                         'ae+embed+classify', 'gan', 'eval']:
            example = example[:2]
        elif loss_type in ["trip"]:
            example = example[:3]
        
        flip = int(random.random() < .5)
        rotate = int(random.random() < .5)
        
        for idx, path in enumerate(example):
            is_sketch = idx == 0    
            images[idx].append(preprocess(Image.open(path), is_sketch, img_size, flip, rotate).to(device))
            
    all_images = [] 
    for img_set in images:
        all_images += img_set
    return torch.cat(all_images)
        
    
def img_path_to_tensor(img_path, is_sketch, img_size=256):
    """ Reads an image and converts to a tensor.
    
    Args:
        img_path: path to the image
        img_size: final size of image
    """
    # Need to adjust this to have .jpg ext
    img = preprocess(Image.open(img_path), is_sketch, img_size)
    return img


def feats_from_img(model, device, img_path, is_sketch, img_size=256):
    """ Converts a list of file paths into corresponding list of tensors.
    
    Args:
        img_list: list of file paths containing images to convert to tensor
        device: cuda or cpu
        img_size: size of image to use
        model: model to use as a feature extractor
    """
    
    tensor = img_path_to_tensor(img_path, is_sketch, img_size).to(device)
    feats = model.extract_features(tensor).detach().cpu().numpy()
    return feats

        
def preprocess(img, is_sketch, img_size=256, flip=False, rotate=False):
    mean = SKETCH_MEAN if is_sketch else PHOTO_MEAN
    std = SKETCH_STD if is_sketch else PHOTO_STD
    
    transforms = []

    if flip:
        transforms += [T.RandomHorizontalFlip(1)]
        
    if not is_sketch:
        transforms += [T.Grayscale(num_output_channels=3)]
    
    transforms += [T.Resize(img_size), 
                   T.ToTensor(), 
                   T.Normalize(mean=torch.tensor(mean, dtype=torch.float32),
                    std=torch.tensor(std, dtype=torch.float32)),
                   T.Lambda(lambda x: x[None])]
    
    transforms = T.Compose(transforms)
    transformed_img = transforms(img)
    if is_sketch:
        img.save('sketch.png')
        tvutils.save_image(transformed_img, 'sketch_preprocess.png')
    else:
        img.save('photo.png')
        tvutils.save_image(transformed_img, 'photo_preprocess.png')
    return transformed_img


def get_img_list(args):
    """ Returns a list containing all the images in the args.phase set. """
    name = args.phase + 'set'
    file = f'/Users/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/{name}.txt' if args.local\
           else f'/home/robincheong/data/sketchy/{name}.txt'
    with open(file, 'r') as f:
        img_list = [c.rstrip() for c in f.readlines()]
    return img_list


def deprocess(img, is_sketch):
    mean = SKETCH_MEAN if is_sketch else PHOTO_MEAN
    std = SKETCH_STD if is_sketch else PHOTO_STD
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=mean, std=[1.0 / s for s in std]),
        T.Normalize(mean=[-m for m in mean], std=std),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)


def get_default_parser():
    """ Parses args for running the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='checkpoint path of the model to load, if '' starts from scratch')
    parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for the DataLoader.')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of img to use')
    parser.add_argument('--dataset', type=str, choices=('eitz', 'sketchy'), default="sketchy", help='which dataset to use')
    parser.add_argument('--local', action="store_true", default=False, help='true if running on local computer')
    parser.add_argument('--model', type=str, required=True, choices=('resnet', 'squeezenet', 'ConvDualVAE', 
                                                                     'ConvDualAE', 'ConvSingleVAE', 'ConvSingleAE',
                                                                     'EmbedGAN'),
                        help='which model to use')
    parser.add_argument('--verbose', action="store_true", default=False, help='turn on for error printing')
    
    
    return parser

def get_train_parser():
    parser = get_default_parser()
    parser.add_argument('--alpha', required=True, type=float, help='changes various loss weightings depending on model type')
    parser.add_argument('--lr', default=1e-3, type=float, help="learning rate to start with")
    parser.add_argument('--wd', default=0, type=float, help="l2 reg term")
    parser.add_argument('--loss_type', type=str, required=True, choices=('classify', 'binary', 
                                                                         'trip', 'quad', 
                                                                         'vae', 'vae+embed', 'vae+embed+classify',
                                                                         'ae', 'ae+embed', 'ae+embed+classify',
                                                                         'gan'),
                        help='which type of contrastive loss to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--toy', action='store_true', help='Use reduced dataset if true.')
    parser.add_argument('--toy_size', type=int, default=5,
                        help='how many of each type to include in the toy dataset.')
    parser.add_argument('--save_dir', type=str, default='/home/robincheong/sketch2img/ckpts/',
                        help='directory in which to save checkpoints.')
    parser.add_argument('--log_dir', type=str, default="logs/",
                        help="directory to save the tensorboard log files to")
    parser.add_argument('--name', type=str, required=True, help='name to use for tensorboard logging')
    parser.add_argument('--h_size', type=int, default=512, help="number of hidden units to use in the AE")
    parser.add_argument('--ftr_extractor_path', type=str, default='', 
                        help='path to model to use as a feature extractor in the AE')
    parser.add_argument('--train_decoders', type=bool, default=False, 
                        help='train decoders without affecting encoders')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--modality', type=str, default='both',
                        help="sketch or photo (for SingleVAE/ SingleAE purposes only")
    parser.add_argument('--photo_encoder_path', type=str)
    parser.add_argument('--sketch_encoder_path', type=str)
    return parser

def get_eval_parser():
    parser = get_default_parser()
    parser.add_argument('--phase', type=str, choices=('train', 'val', 'test'), default='val', 
                        help="x set to evaluate over")
    parser.add_argument('--h_size', type=int, default=512)
    parser.add_argument('--ftr_extractor_path', type=str, default='', 
                        help='path to model to use as a feature extractor in the AE')
    parser.add_argument('--photo_encoder_path', type=str)
    parser.add_argument('--sketch_encoder_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--loss_type', type=str, default='eval')
    parser.add_argument('--num_fails', type=int, default=5)
    parser.add_argument('--num_success', type=int, default=5)
    return parser
    
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
