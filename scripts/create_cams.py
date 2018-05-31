from __future__ import print_function, division

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.models import SqueezeNet, ResNet
from matplotlib.image import imread
from PIL import Image

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from pathlib import Path
import copy
import cv2
from utils import get_default_parser, preprocess

# CITE

def returnCAM(feature_conv, weight_softmax, class_idx):
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

def create_cams(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)

    model.to(device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    features = []

    def hook_feature(module, input_, output):
        features.append(output.data.cpu().numpy())

    model._modules.get('features')[7][1].register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    
    examples_csv = pd.read_csv("/home/robincheong/data/sketchy/{}set.csv".format(args.phase)))
    classes = {}
    for i, row in examples_csv.iterrows():
        classes[row['Label']] = row['Photo Path'].split('/')[7]

    examples = examples_csv.sample(args.num_cams)
    for i, example in enumerate(examples):
        photo_path = example['Photo Path']
        sketch_path = example['Sketch Path']
        cor_class_idx = example['Label']
        catg = example['Photo Path'].split('/')[7]
        
        # compute class scores
        sketch = preprocess(Image.open(sketch_path), is_sketch=True).unsqueeze(0)
        sketch_logits = model(sketch.to(device))
        
        photo = preprocess(Image.open(photo_path), is_sketch=False).unsqueeze(0)
        photo_logits = model(photo.to(device))

        # compute probabilites and sort
        sketch_probs = F.softmax(sketch_logits, dim=1).cpu().data.squeeze()
        sketch_probs, sketch_idx = sketch_probs.sort(0, True)
        sketch_probs, sketch_idx = sketch_probs.numpy(), sketch_idx.numpy()
        
        cor_class_idx = idx[0]
        cor_class = classes[cor_class_idx]
        
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(sketch_probs[i], classes[sketch_idx[i]]))

        photo_probs = F.softmax(photo_logits, dim=1).cpu().data.squeeze()
        photo_probs, photo_idx = photo_probs.sort(0, True)
        photo_probs, photo_idx = photo_probs.numpy(), photo_idx.numpy()

        for i in range(0, 5):
            print('{:.3f} -> {}'.format(photo_probs[i], classes[photo_idx[i]]))
        
        # generate class activation mapping for the top1 prediction
        sketch_CAMs = returnCAM(features[i], weight_softmax, [idx[0]])
        photo_CAMs = returnCAM(features[i+1], weight_softmax, [idx[0]])

        # render the CAM and output
        for modality, path in [("sketch", sketch_path), ("photo", photo_path)]:
            print('Rendering {} CAMs for the correct class: {}'.format(modality, cor_class))
            img = cv2.imread(str(path))
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite('cams/sketch_{}_classify+binary_only.jpg'.format(cat), result)
        
        
if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--phase', type=str, choices=('train', 'val', 'test'), default='val', 
                        help="x set to evaluate over")
    parser.add_argument('--num_cams', type=int, default=1,
                        help="number of cams to generate")
    args = parser.parse_args()
    eval_model(args)
                          

