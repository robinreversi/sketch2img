import numpy as np
from torch.utils.data import Dataset


class SketchyDataset(Dataset):

    def __init__(self, photo_transform="tx_000000000000/", sketch_transform="tx_000000000000/", split="train"):
        """
        Args:
            transform: The type of image transformation to use as specified in the README
            split: If train, shuffle pairs and define len as max of len(src_paths) and len(tgt_paths).
                   If not train, take pairs in order and define len as len(src_paths).
        """
        
        PARENTDIR = "/home/robincheong/data/sketchy/"
        data_file = self._load_file(PARENTDIR + split + ".txt")
        
        photo_dir = PARENTDIR + f"photo/{photo_transform}"
        sketch_dir = PARENTDIR + f"photo/{sketch_transform}"
        
        self.img_format = img_format
        self.is_training = is_training

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def _load_data(filepath):
        data = {}
        with open(filepath, "r") as fp:
            