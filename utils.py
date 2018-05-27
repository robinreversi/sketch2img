from loaders.EitzDataLoader import EitzDataLoader
from loaders.SketchyDataLoader import SketchyDataLoader
from PIL import Image
import numpy as np
import argparse
import torch
import torchvision.transforms as T


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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


def load_sketchy_images(inputs, loss_type, device, img_size=512):
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
            images[idx].append(preprocess(Image.open(path), img_size).to(device))
    all_images = [] 
    for img_set in images:
        all_images += img_set
    return torch.cat(all_images)
        
def get_loss_fn(dataset, loss_type):
    if dataset == "eitz" or loss_type == "classify":
        return nn.CrossEntropyLoss()
    elif dataset == "sketchy":
        def mse_loss(input_, target):
            return torch.sum((input_ - target) ** 2) / len(input_)

        if loss_type == "binary":
            return mse_loss

        elif loss_type == "trip":
            def trip_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, alpha=.2):
                loss1 = mse_loss(sketch_embed, correct_photo_embed)
                loss2 = mse_loss(sketch_embed, same_cat_diff_photo_embed)
                total_loss = max(loss1 - loss2 + alpha, 0)
                return sum(total_loss)

            return trip_loss

        elif loss_type == "quad":
            def quad_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, diff_cat_photo_embed, alpha=.2):
                loss = nn.MSELoss()
                loss1 = max(loss(sketch_embed, correct_photo_embed) 
                            - loss(sketch_embed, same_cat_diff_photo_embed) + alpha, 0)
                loss2 = max(loss(sketch_embed, correct_photo_embed),
                            - loss(sketch_embed, diff_cat_photo_embed) + alpha, 0)
                loss3 = max(loss(sketch_embed, same_cat_diff_photo_embed)
                            - loss(sketch_embed, diff_cat_photo_embed) + alpha, 0)
                total_loss = loss1 + loss2 + loss3
                return sum(total_loss)

            return quad_loss
    else: 
        raise ValueError
    
def img_path_to_tensor(img_path, img_size=512):
    """ Reads an image and converts to a tensor.
    
    Args:
        img_path: path to the image
        img_size: final size of image
    """
    img = preprocess(Image.open(img_path), img_size)
    return img


def feats_from_img(model, device, img_path, img_size=512):
    """ Converts a list of file paths into corresponding list of tensors.
    
    Args:
        img_list: list of file paths containing images to convert to tensor
        device: cuda or cpu
        img_size: size of image to use
        model: model to use as a feature extractor
    """
    
    tensor = img_path_to_tensor(img_path, img_size).to(device)
    feats = model.extract_features(tensor).detach().cpu().numpy()
    return feats

        
def preprocess(img, size=512, mean=[0,0,0], std=[1,1,1]):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean,
                    std=std),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def get_img_list(phase):
    """ Returns a list containing all the images in the test set.
    
    Args:
        phase: the phase of development -- one of (train / val / test)
    """
    name = phase + 'set'
    file = f'/Users/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/{name}.txt'
#     file = f'/home/robincheong/data/sketchy/{name}.txt'
    with open(file, 'r') as f:
        img_list = [c.rstrip() for c in f.readlines()]
    return img_list


def deprocess(img, mean=[0,0,0], std=[1,1,1]):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=mean, std=[1.0 / s for s in std]),
        T.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
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
    parser.add_argument('--num_epochs', type=int, default=10, help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--checkpoints_dir', type=str, default='/home/robincheong/sketch2img/checkpoints/',
                        help='Directory in which to save checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to checkpoint to load. If empty, start from scratch.')
    parser.add_argument('--save_dir', type=str, default='/home/robincheong/sketch2img/model_states/',
                        help='Path to save directory')
    parser.add_argument('--name', type=str, help='Experiment name.')
    parser.add_argument('--img_format', type=str, default='png', choices=('jpg', 'png'), help='Format for input images')
    parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for the DataLoader.')
    parser.add_argument('--toy', action='store_true', help='Use reduced dataset if true.')
    parser.add_argument('--toy_size', type=int, default=5,
                        help='How many of each type to include in the toy dataset.')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Size of img to use')
    parser.add_argument('--dataset', type=str, required=True, choices=('eitz', 'sketchy'), help='which dataset to use')
    parser.add_argument('--local', required=True, help='true if running on local computer')
    parser.add_argument('--loss_type', type=str, required=True, choices=('classify', 'binary', 'trip', 'quad'), 
                        help='which type of contrastive loss to use')
    return parser

