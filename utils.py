from loaders.EitzDataLoader import EitzDataLoader
from PIL import Image
import numpy as np
import argparse
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
        dataloaders = {'train': SketchyDataLoader(args.n_workers, args.batch_size, 'train'), 
                       'val': SketchyDataLoader(args.n_workers, 1, 'val'), 
                       'test': SketchyDataLoader(args.n_workers, 1, 'test')}
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    return dataloaders

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
    with open(f'/home/robincheong/data/sketchy/{name}.txt','r') as f:
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
    return parser

