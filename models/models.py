from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from .archetypes import *
    
        
class SqueezeNet(FeatureExtractor):
    """ SqueezeNet is a pre-trained model to be fine-tuned on the Eitz 2012 Dataset as a feature extractor
        for both sketches and photos.
    """
    def __init__(self, args):
        """
        Args:
            args: Configuration args passed in via the command line.
        """
        super().__init__()
        self.features = torchvision.models.squeezenet1_1(pretrained=True).features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.name = 'squeeze_net'

        if args.dataset == 'eitz':
            num_labels = 250
        elif args.dataset == "sketchy":
            num_labels = 125
        else:
            raise NotImplementedError
            
        self.classifier = nn.Linear(512, num_labels)

        
    def forward(self, x):
        """ Performs a forward pass of the model.

        Args:
            x: the image to compute a forward pass on
        """
        N, H, W = x.shape
        x = x.unsqueeze(1).expand(N, 3, H, W)
        logits = self.features(x)
        logits = self.gap(logits)
        logits = logits.view(logits.size(0), -1)
        logits = self.classifier(logits)
        return logits
    
    
    def extract_features(self, x, is_sketch=True):
        """ Uses the SqueezeNet model to extract features from x.

        Args:
            x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
                  will be fed to the SqueezeNet.
            is_sketch: No-op since SqueezeNet extracts the same features regardless 

        Returns:
            features: The activations for each PyTorch Tensor after the GAP layer as a Tensor
        """
        features = self.features(x)
        features = self.gap(features)
        return features

    
class ResNet(FeatureExtractor):
    
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 125)
        self.name = "resnet"

    def forward(self, x):
        logits = self.extract_features(x)
        logits = self.classifier(logits)
        return logits
    
    def extract_features(self, x):
        N = len(x)
        features = self.features(x)
        features = self.gap(features).squeeze()
        features = features.view(N, -1)
        return features
    
    def make_predictions(self, features):
        return self.classifier(features)

    
class ConvUpscaleBlock(nn.Module):
    """ Returns a upscaling block that upscales, applies relu, and batchnorms. """
    def __init__(self, in_feats, out_feats, k_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_feats, out_feats, k_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_feats)
        )
    
    def forward(self, x):
        return self.block(x)
    
    
class ConvDecoder(Decoder):
    
    def __init__(self, args):
        super().__init__()
        
        # CITE code inspired by CS231n A3 GANs
        
        # takes in vector of size 512 
        # outputs an image of size 256 (scaled to be between -1 and 1)
        
        self.decoder = nn.Sequential(
            nn.Linear(args.h_size, 8 * 8 * 64), # 2.1 mil parameters
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 64),
            # starts at W, H = 8, 8
            Unflatten(N=-1, C=64, W=8, H=8),
            ConvUpscaleBlock(64, 32, 4, 2, 1), # after 16, 32k parameters
            ConvUpscaleBlock(32, 16, 4, 2, 1), # after 32, 8k parameters
            ConvUpscaleBlock(16, 8, 4, 2, 1), # after 64, 2k parameters
            ConvUpscaleBlock(8, 4, 4, 2, 1), # after 128, .5k parameters,
            nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1), # after 256, .125k parameters 
            nn.Tanh()
        )                            
    def decode(self, x):    
        return self.decoder(x)

    def forward(self, x):
        return self.decode(x)

    
class ConvEncoder(nn.Module):
    
    def __init__(self, args, ftr_extractor=None):
        super().__init__()
        if ftr_extractor:
            self.ftr_extractor = ftr_extractor 
        else:
            self.ftr_extractor = ResNet()
        self.fc_mean = nn.Linear(512, args.h_size)
        self.fc_logvar = nn.Linear(512, args.h_size)
    
    def forward(self, x):
        a = self.ftr_extractor.extract_features(x)
        return self.fc_mean(a), self.fc_logvar(a)
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
        
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        
   
class ConvVAE(FeatureExtractor):
    
    def __init__(self, args, ftr_extractor=None):
        super().__init__()
        
        self.encoder = ConvEncoder(args, ftr_extractor)
        self.photo_decoder = ConvDecoder(args)
        self.sketch_decoder = ConvDecoder(args)
    
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z, is_sketch):
        """ Reparameterized version of the encoding. """
        return self.sketch_decoder(z) if is_sketch else self.photo_decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x, is_sketch):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decode_out = self.decode(z, is_sketch)
        return decode_out, mu, logvar
