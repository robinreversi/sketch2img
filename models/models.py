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

class SqueezeNet(nn.Module):
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
        self.classifier = nn.Linear(512, args.num_labels)

    def forward(self, x):
        N, H, W = x.shape
        x = x.unsqueeze(1).expand(N, 3, H, W)
        logits = self.features(x)
        logits = self.gap(logits)
        logits = logits.view(logits.size(0), -1)
        logits = self.classifier(logits)
        return logits