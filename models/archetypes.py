from __future__ import print_function, division

import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError

    def extract_features(self, x):
        raise NotImplementedError

class Encoder(FeatureExtractor):
    def __init__(self):
        super().__init__()
    