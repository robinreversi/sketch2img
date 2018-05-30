import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset


class SketchyDataset(Dataset):
    
    def __init__(self, args, phase):
        """
        Args:
            local: whether I'm running this code on my local computer or gcloud
            phase: One of "train", "val", "test"
            loss_type: One of "binary", "trip", "quad" 
        """
        self.data_dir = "/Users/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/" \
                        if args.local  else "/home/robincheong/data/sketchy/" 
        self.phase = phase
        
        self.data = pd.read_csv(os.path.join(self.data_dir, "{}set.csv".format(phase)))
        
        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, item):
        selection = self.data.iloc[item]
        correct_photo_path = selection['Photo Path']
        sketch_path = selection['Sketch Path']
        label = selection['Label']
                
        same_category = self.data[(self.data['Label'] == label) 
                                  & (self.data['Sketch Path'] != sketch_path)
                                  & (self.data['Photo Path'] != correct_photo_path)]
        same_cat_diff_photo_path = same_category.sample()['Sketch Path'].item()

        diff_category = self.data[(self.data['Label'] != label)]
        diff_cat_photo_path = diff_category.sample()['Sketch Path'].item()
        return '++'.join([sketch_path, correct_photo_path, same_cat_diff_photo_path, diff_cat_photo_path]), label
        