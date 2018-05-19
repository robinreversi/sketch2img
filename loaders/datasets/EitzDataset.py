import numpy as np
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .constants import *

class EitzDataset(Dataset):
    """Base dataset for CT studies."""

    def __init__(self, args, phase):
        """
        Args:
            phase: The phase to use = (train, val, test)
        """
        self.is_training = phase == 'train'
        self.img_size = args.img_size
        self.data = self._load_data(phase, args.toy, args.toy_size)


    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = self._get_img(img_path)
        return img, label

    
    def __len__(self):
        return len(self.data)
    
    def _get_img(self, img_path):
        """Load an image from `img_path`."""
        img = Image.open(EITZ_FP + "sketches/" + img_path)
        img.thumbnail((self.img_size, self.img_size), Image.ANTIALIAS)
        return np.array(img)
    
    def _load_data(self, phase, toy=False, toy_size=False):
        """Loads data from `phase` set
        
        Args:
            phase: The phase of development = (train, val, test)
            toy: If true, only use toy_size samples
            toy_size: number of samples to use if toy is true
        """
        csv_name = phase + ".csv"
        data = pd.read_csv(EITZ_FP + csv_name)
        if toy:
            data = data.sample(toy_size)
        data = list(data.itertuples(index=False))
        return data
            