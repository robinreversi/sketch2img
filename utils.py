from loaders.EitzDataLoader import EitzDataLoader
from loaders.SketchyDataLoader import SketchyDataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
import argparse
import torch
import torchvision.transforms as T


PHOTO_MEAN = np.array([0.47122188, 0.44775212, 0.39636577], dtype=np.float32)
SKETCH_MEAN = np.array([0.95263444, 0.95263444, 0.95263444], dtype=np.float32)

SKETCH_STD = np.array([0.35874852, 0.35874852, 0.35874852], dtype=np.float32)
PHOTO_STD = np.array([0.46127741, 0.46127741, 0.46127741], dtype=np.float32)

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
        loss_type: the type of loss to use, one of "classify", "binary", "trip", "quad"
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
        if loss_type == "classify" or loss_type == "binary":
            example = example[:2]
        elif loss_type == "trip":
            example = example[:3]
        for idx, path in enumerate(example):
            is_sketch = idx == 0
            images[idx].append(preprocess(Image.open(path), is_sketch, img_size).to(device))
    all_images = [] 
    for img_set in images:
        all_images += img_set
    return torch.cat(all_images)
        
def get_loss_fn(dataset, loss_type):
    ce_loss = nn.CrossEntropyLoss()
    
    def mse_loss(input_, target):
        return (input_ - target) ** 2 / len(input_)
    
    if dataset == "eitz" or loss_type == "classify":
        def classify_loss(sketch_logits, photo_logits, labels):
            return ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)

        return classify_loss
    elif dataset == "sketchy":
        
        if loss_type == "binary":
            def binary_loss(sketch_embed, correct_photo_embed, 
                            sketch_logits, photo_logits, labels):
                
                embedding_loss = torch.sum(mse_loss(sketch_embed, correct_photo_embed))
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 
                
            return binary_loss

        elif loss_type == "trip":
            def trip_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, 
                          sketch_logits, photo_logits, labels, alpha=.2):
                
                loss1 = mse_loss(sketch_embed, correct_photo_embed)
                loss2 = mse_loss(sketch_embed, same_cat_diff_photo_embed)
                embedding_loss = torch.sum(torch.clamp(loss1 - loss2 + alpha, min=0))
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 
            
            return trip_loss

        elif loss_type == "quad":
            def quad_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, diff_cat_photo_embed, 
                          sketch_logits, photo_logits, labels, alpha=.2):
                
                loss1 = torch.clamp(mse_loss(sketch_embed, correct_photo_embed) 
                            - mse_loss(sketch_embed, same_cat_diff_photo_embed) + alpha, min=0)
                loss2 = torch.clamp(mse_loss(sketch_embed, correct_photo_embed)
                            - mse_loss(sketch_embed, diff_cat_photo_embed) + alpha, min=0)
                loss3 = torch.clamp(mse_loss(sketch_embed, same_cat_diff_photo_embed)
                            - mse_loss(sketch_embed, diff_cat_photo_embed) + alpha, min=0)
                embedding_loss = torch.sum(loss1 + loss2 + loss3)
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 

            return quad_loss
    else: 
        raise ValueError
    
def img_path_to_tensor(img_path, is_sketch, img_size=512):
    """ Reads an image and converts to a tensor.
    
    Args:
        img_path: path to the image
        img_size: final size of image
    """
    # Need to adjust this to have .jpg ext
    img = preprocess(Image.open(img_path), is_sketch, img_size)
    return img


def feats_from_img(model, device, img_path, is_sketch, img_size=512):
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

        
def preprocess(img, is_sketch, img_size=256):
    mean = SKETCH_MEAN if is_sketch else PHOTO_MEAN
    std = SKETCH_STD if is_sketch else PHOTO_STD
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean, dtype=torch.float32),
                    std=torch.tensor(std, dtype=torch.float32)),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


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


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def get_default_parser():
    """ Parses args for running the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='checkpoint path of the model to load, if '' starts from scratch')
    parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for the DataLoader.')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Size of img to use')
    parser.add_argument('--dataset', type=str, required=True, choices=('eitz', 'sketchy'), help='which dataset to use')
    parser.add_argument('--local', action="store_true", default=False, help='true if running on local computer')
    parser.add_argument('--model', type=str, required=True, choices=('resnet', 'squeezenet'), help='which model to use')
    
    return parser

