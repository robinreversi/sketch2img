from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle
sys.path.append('/home/robincheong/sketch2img/')

from models import SqueezeNet, ResNet
from matplotlib.image import imread
from PIL import Image


from pathlib import Path
import cv2
from utils import get_default_parser, preprocess, get_dataloaders, load_sketchy_images
from model_utils import load_model

# CITE

def return_CAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    c, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



def get_probs_and_idx(imgs, model, device, is_sketch):
    """ Computes forward pass on img and gets sorted probs for each class.
    
    Args:
        img_path: path to the image
        model: model to use to compute forward pass
        device: cuda / cpu
        is_sketch: is the image a sketch? if so, normalize differently
        
    Returns:
        img_probs: sorted list of class probabilties
        img_idx: sorted list of class idxs (int representation of class)
        
    """
    img_logits = model(imgs)
    img_probs = F.softmax(img_logits, dim=1).cpu().data.squeeze()
    img_probs, img_idx = img_probs.sort(1, True)
    img_probs, img_idx = img_probs.numpy(), img_idx.numpy()
    
    return img_probs, img_idx


def print_top_5(probs, idx, classes, cor_class_idx, modality):
    print("Top 5 predictions for {}:".format(modality))
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    print("GT: {} -> {}".format(probs[np.where(idx == cor_class_idx)], classes[cor_class_idx]))
    print("=" * 20)

    
def set_up_model(model):
    features = []

    def hook_feature(module, input_, output):
        features.append(output.data.cpu().numpy())

    # TODO: Only works with resnet architecture
    model._modules.get('features')[7][1].register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    
    return features, weight_softmax


def create_cams(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(args, device)

    features, weight_softmax = set_up_model(model)
    
    with open("/home/robincheong/data/sketchy/idx_to_class_dict.pkl", "rb") as f:
        classes = pickle.load(f)
        
    loader = get_dataloaders(args)[args.phase]
    
    num_cams = 0
    
    for inputs, labels in loader:
        print("Getting logits")
        labels = labels.numpy()
        
        file_paths = [example.split('++') for example in inputs]
        
        N = len(inputs)
        inputs = load_sketchy_images(inputs, args.loss_type, device, args.img_size)
        sketches, photos = torch.split(inputs, N)
        
        sketch_probs, sketch_idx = get_probs_and_idx(sketches, model, device, is_sketch=True)
        photo_probs, photo_idx = get_probs_and_idx(photos, model, device, is_sketch=False)
        
        print(sketch_probs.shape)
            
        print("Generating CAMs")
        

        
        for i in range(N):
            if num_cams > args.num_cams:
                break
            num_cams += 1
            print_top_5(sketch_probs[i], sketch_idx[i], classes, labels[i], "sketch")
            print_top_5(photo_probs[i], photo_idx[i], classes, labels[i], "photo")
            CAMs = {"sketch": return_CAM(features[0][i], weight_softmax, [sketch_idx[i][np.where(sketch_idx[i] == labels[i])]]),
                    "photo": return_CAM(features[1][i], weight_softmax, [photo_idx[i][np.where(photo_idx[i] == labels[i])]])}

            # render the CAM and output
            for modality, path in [("sketch", file_paths[i][0]), ("photo", file_paths[i][1])]:
                print('Rendering {} CAMs for the correct class: {}'.format(modality, classes[labels[i]]))
                img = cv2.imread(str(path))
                height, width, _ = img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[modality][0],(width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.3 + img * 0.5

                cam_fname = 'cams/{}_{}{}.jpg'.format(modality, classes[labels[i]], args.suffix)

                cv2.imwrite(cam_fname, result)
            
        break
        
        
        
if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--phase', type=str, choices=('train', 'val', 'test'), default='val', 
                        help="x set to evaluate over")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss_type', type=str, default='eval')
    parser.add_argument('--num_cams', type=int, default=1,
                        help="number of cams to generate")
    parser.add_argument('--suffix', type=str, default='',
                        help="suffix for generated CAMs (use something like 'bin_loss')")
    parser.add_argument('--random_seed', type=int, default=0,
                        help="seed for sampling")
    args = parser.parse_args()
    create_cams(args)
                          

