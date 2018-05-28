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
    
    def __init__(self, args):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512, 125)
        self.name = "resnet"

    def forward(self, x):
        logits = self.features(x)
        logits = self.classifier(logits)
        return logits
    
    def extract_features(self, x):
        return self.features(x)
    
    def make_prediction(self, features):
        return self.classifier(features)
    
# [WIP]
class VAE(FeatureExtractor):
    
    def __init__(self, args):
        
        # encoder portion of the network uses 
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_mean = nn.Linear(512, args.h_size)
        self.fc_logvar = nn.Linear(512, args.h_size)
        
    def encode(self, x):
        a = self.features(x)
        a = self.gap(a)
        return self.fc_mean(a), self.fc_logvar(a)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
