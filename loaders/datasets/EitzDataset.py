import numpy as np
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .constants import *

class EitzDataset(Dataset):
    """Base dataset for CT studies."""

    def __init__(self, split, img_size=256):
        """
        Args:
            split: The split to use = (train, val, test)
        """
        self.is_training = split == 'train'
        self.img_size = img_size
        self.data = self._load_data(split)


    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = self._get_img(img_path)
        return img, label

    
    def __len__(self):
        return len(self.data)
    
    def _get_img(self, img_path):
        """Load an image from `img_path`. Use format `self.img_format`."""
        img = Image.open(EITZ_FP + "sketches/" + img_path)
        img.thumbnail((self.img_size, self.img_size), Image.ANTIALIAS)
        return np.array(img)
    
    def _load_data(self, split):
        """Loads data from `split` set"""
        csv_name = split + ".csv"
        data = pd.read_csv(EITZ_FP + csv_name)
        data = list(data.itertuples(index=False))
        return data
            