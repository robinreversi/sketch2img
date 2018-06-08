from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as tvutils

from collections import defaultdict, namedtuple
from scipy.misc import imread
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import copy
from pathlib import Path
from PIL import Image
import cv2
import random


from models.models import SqueezeNet, ResNet, ConvDualVAE, ConvDualAE, ConvSingleAE, ConvSingleVAE
from utils import get_dataloaders, get_default_parser, load_sketchy_images, log_metrics, get_eval_parser

from model_utils import get_loss_fn, load_model, vae_forward, ae_forward, \
                        classify_contrast_forward, gan_forward

import argparse
import pickle
from loaders.datasets.constants import *




def save_knns(sketch_path, knns, idx2photo, name, img_loc):
    comb_imgs = [np.array(Image.open(sketch_path))] 
    comb_imgs += [np.array(Image.open(idx2photo[idx])) for idx in knns] 
    cv2.rectangle(comb_imgs[0], (0, 0), (255, 255), (0, 0, 255), 10)
    if img_loc != -1:
        cv2.rectangle(comb_imgs[img_loc + 1], (0, 0), (255, 255), (0, 255, 0), 10)
        print(comb_imgs[img_loc+1].shape)
    comb_imgs = np.hstack(comb_imgs)
    comb_imgs = Image.fromarray(comb_imgs)
    

    comb_imgs.save(name)
    
def eval_model(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args, device)
    model.eval()

    loader = get_dataloaders(args)[args.phase]

    with open('photo2idx_{}.pkl'.format(args.phase), 'rb') as f:
        photo2idx = pickle.load(f)
    idx2photo = {i: photo2idx[i] for i in photo2idx}

    test_ftrs = np.zeros((len(photo2idx), 512))
    train_ftrs = defaultdict(list)

    batch_num = 0 

    for inputs, _ in loader:
        print("Batch num: ", batch_num)
        batch_num += 1
        N = len(inputs)
        photo_idxs = [photo2idx[example.split('++')[1]] for example in inputs]
        sketch_paths = [example.split('++')[0] for example in inputs]
        inputs = load_sketchy_images(inputs, args.loss_type, device, args.img_size)
        features = model.extract_features(inputs).cpu().detach()
        indices = torch.tensor(range(0, 2 * N))
        selected_features = torch.index_select(features, 0, indices)
        sketch_ftrs, photo_ftrs = torch.split(selected_features, N)

        for i, photo_idx in enumerate(photo_idxs):
            test_ftrs[photo_idx] = photo_ftrs[i].numpy()
            train_ftrs[photo_idx].append((sketch_ftrs[i].numpy(), sketch_paths[i]))

        del inputs
        del features
    
    idx2photo = {photo2idx[path]: path for path in photo2idx}
    
    nbrs = NearestNeighbors(n_neighbors=len(test_ftrs), algorithm='brute', metric='l2').fit(test_ftrs)

    top_5 = 0
    top_1 = 0
    total_imgs = 0
    fails = 0
    successes = 0

    # frequency dist. of the rank the true matching image is at
    # ranking_hist = [0 for _ in range(1250)]

    print("Evaluating photos")

    for test_img_idx in train_ftrs:
        print(f"EVALUATING PHOTO NO {test_img_idx} / 1250")
        total_imgs += len(train_ftrs[test_img_idx])
        for sketch_feat, sketch_path in train_ftrs[test_img_idx]:
            distances, knns = nbrs.kneighbors(sketch_feat.reshape(1, -1), n_neighbors=5)
            knns = knns[0]
            if test_img_idx in knns:
                top_5 += 1
                if successes < 10 and random.random() < .1:
                    successes += 1
                    img_loc = np.where(knns == test_img_idx)[0][0]
                    print(img_loc)
                    save_knns(sketch_path, knns, idx2photo, "search/success_{}.png".format(successes), img_loc)

            elif fails < 10 and random.random() < .1:
                fails += 1
                save_knns(sketch_path, knns, idx2photo, "search/fails_{}.png".format(fails), -1)

            if test_img_idx == knns[0]:
                top_1 += 1



            for idx, neighbor in enumerate(knns):
                if neighbor == test_img_idx:
                    print(f'\t ranking for img {test_img_idx} found at {idx + 1}')
                    break


    print(f'Total imgs: {total_imgs}')
    print(f'Total Top 1 Correct: {top_1}')
    print(f'Total Top 5 Correct: {top_5}')
    print(f'top_1 acc = {top_1 / total_imgs}')
    print(f'top_5 acc = {top_5 / total_imgs}')


if __name__ == '__main__':
    parser = get_eval_parser()
    args = parser.parse_args()
    eval_model(args)
                          




