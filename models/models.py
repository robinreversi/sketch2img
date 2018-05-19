from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError

    def extract_features(self, x, is_sketch=True):
        raise NotImplementedError

class SqueezeNet(FeatureExtractor):
    """
        SqueezeNet is a pre-trained model designed to be fine tuned on the Eitz 2012 Dataset
        for better performance on SBIR
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