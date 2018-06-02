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
from utils import get_default_parser, preprocess

# CITE

def return_CAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    b, c, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((c, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def load_model(args, device):
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)

    model.to(device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    
    return model


def get_probs_and_idx(img_path, model, device, is_sketch):
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
    img = preprocess(Image.open(img_path), is_sketch=is_sketch)
    img_logits = model(img.to(device))

    img_probs = F.softmax(img_logits, dim=1).cpu().data.squeeze()
    img_probs, img_idx = img_probs.sort(0, True)
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
    
    examples_csv = pd.read_csv("/home/robincheong/data/sketchy/{}set.csv".format(args.phase))
    
    with open("/home/robincheong/data/sketchy/idx_to_class_dict.pkl", "rb") as f:
        classes = pickle.load(f)
        
    examples = examples_csv.sample(args.num_cams, random_state=args.random_seed)
    for j in range(len(examples)):
        example = examples.iloc[j]
        photo_path = example['Photo Path']
        sketch_path = example['Sketch Path']

        cor_class_idx = int(example['Label'])
        cor_class = classes[cor_class_idx]
        
        sketch_probs, sketch_idx = get_probs_and_idx(sketch_path, model, device, is_sketch=True)
        print_top_5(sketch_probs, sketch_idx, classes, cor_class_idx, "sketch")
        
        photo_probs, photo_idx = get_probs_and_idx(sketch_path, model, device, is_sketch=False)
        print_top_5(photo_probs, photo_idx, classes, cor_class_idx, "photo")
        
        # generate class activation mapping for the gt label
        CAMs = {"sketch": return_CAM(features[j], weight_softmax, [sketch_idx[np.where(sketch_idx == cor_class_idx)]]),
                "photo": return_CAM(features[j+1], weight_softmax, [photo_idx[np.where(photo_idx == cor_class_idx)]])}

        # render the CAM and output
        for modality, path in [("sketch", sketch_path), ("photo", photo_path)]:
            print('Rendering {} CAMs for the correct class: {}'.format(modality, cor_class))
            img = cv2.imread(str(path))
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[modality][0],(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            
            cam_fname = 'cams/{}_{}{}.jpg'.format(modality, cor_class, args.suffix)
                
            cv2.imwrite(cam_fname, result)
        
        
if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--phase', type=str, choices=('train', 'val', 'test'), default='val', 
                        help="x set to evaluate over")
    parser.add_argument('--num_cams', type=int, default=1,
                        help="number of cams to generate")
    parser.add_argument('--suffix', type=str, default='',
                        help="suffix for generated CAMs (use something like 'bin_loss')")
    parser.add_argument('--random_seed', type=int, default=0,
                        help="seed for sampling")
    args = parser.parse_args()
    create_cams(args)
                          

