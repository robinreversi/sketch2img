import torch
import torch.nn as nn
import torchvision
import PIL
from PIL import Image

from collections import defaultdict
import numpy as np
from loaders.datasets.constants import *
import os
from scipy.misc import imread
from collections import namedtuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,LSHForest
from models.models import SqueezeNet, ResNet
from utils import get_img_list, img_path_to_tensor, feats_from_img, get_default_parser

def eval_model(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)

    model.to(device)
    
    if args.ckpt_path:
        model.load_state_dict(torch.load(args.ckpt_path))
    
    model.eval()
    
    test_img_list = get_img_list(args)
    test_feats = []
        
    # mapping from a photo idx to a list of features of sketches associated 
    # with the photo
    photo2sketch = defaultdict(list)
    
    if args.pca:
        all_sketch_feats = []
    
    print("Generating feats...")
    
    for i, local_path in enumerate(test_img_list):
        img_cat, img_name = local_path.split('/')
        img_name = img_name.split(".")[0]
        full_photo_path = os.path.join(PHOTO_DIR, local_path)
        is_sketch = False
        test_feats.append(feats_from_img(model, device, full_photo_path, is_sketch, args.img_size))
        
        sketches_in_cat = os.listdir(SKETCH_DIR + img_cat)
        matching_sketches = [sketch for sketch in sketches_in_cat if sketch.startswith(img_name)]
        for sketch in matching_sketches:
            full_sketch_path = os.path.join(SKETCH_DIR, img_cat, sketch)
            is_sketch = True
            sketch_feats = feats_from_img(model, device, full_sketch_path, is_sketch, args.img_size)
            sketch_feats = sketch_feats.reshape(sketch_feats.shape[0], -1)
            photo2sketch[i].append(sketch_feats)
            if args.pca:
                all_sketch_feats.append(sketch_feats)
                

    test_feats = np.array(test_feats)
    test_feats = test_feats.reshape(test_feats.shape[0], -1)

    if args.pca:    
        pca = PCA(n_components=args.pca)
        pca.fit(all_sketch_feats)
        test_feats = pca.transform(test_feats)
    
    nbrs = NearestNeighbors(n_neighbors=len(test_feats), algorithm='brute', metric='l2').fit(test_feats)

    top_5 = 0
    top_1 = 0
    total_imgs = 0
    
    # frequency dist. of the rank the true matching image is at
    # ranking_hist = [0 for _ in range(1250)]
        
    for test_img in photo2sketch:
        print(f"EVALUATING PHOTO NO {test_img} / 1250")
        total_imgs += len(photo2sketch[test_img])
        for sketch_feat in photo2sketch[test_img]:
            if args.pca:
                sketch_feat = pca.transform(sketch_feat)
            distances, knns = nbrs.kneighbors(sketch_feat, n_neighbors=5)
            knns = knns[0]
            if test_img in knns:
                top_5 += 1
            if test_img == knns[0]:
                top_1 += 1
                
            for idx, neighbor in enumerate(knns):
                if neighbor == test_img:
                    print(f'\t ranking for img {test_img} found at {idx + 1}')
                    break
 
            
    print(f'Total imgs: {total_imgs}')
    print(f'Total Top 1 Correct: {top_1}')
    print(f'Total Top 5 Correct: {top_5}')
    print(f'top_1 acc = {top_1 / total_imgs}')
    print(f'top_5 acc = {top_5 / total_imgs}')


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--pca', type=int, default=0, 
                        help="Use PCA to reduce dims to specified value; if 0, do not use PCA")
    parser.add_argument('--phase', type=str, choices=('train', 'val', 'test'), default='val', 
                        help="x set to evaluate over")
    args = parser.parse_args()
    eval_model(args)
                          
