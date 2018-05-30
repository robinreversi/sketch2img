import os
from collections import defaultdict
import numpy as np
import pandas as pd
import argparse

def remove_invalids(prefix, file_names, transforms, verbose=False):
    """ Remove bad files from dataset. """
    invalid = set()

    # read in text files and add to a set 
    for file_name in file_names:
        with open(prefix + 'info/' + file_name, 'r') as f:
            files = f.readlines()
            files = map(lambda x: x[:-1] + ".png", files)
            invalid |= set(files)
            
    sketch_datadir = prefix + "sketch/"
    for transform in transforms:
        datadir = sketch_datadir + transform

        for sketchdir in os.listdir(datadir):
            if verbose:
                print(f"Walking through {sketchdir}...")
            for file in os.listdir(datadir + sketchdir):
                if file in invalid:
                    if verbose:
                        print(f"Removing file: {file}")
                    os.remove(datadir + sketchdir + "/" + file)
                    

def make_labels_dict(photos_list):
    """ Constructs a mapping from category name to a label. """
    labels = list(set(map(lambda x: x.split('/')[0], photos_list)))
    labels_dict = {label: val for val, label in enumerate(labels)}
    return labels_dict

def create_sketchy_set(prefix, transform, photos_list, phase, labels_dict):
    """ Creates and saves a phase set for sketchy.
    
    Args:
        prefix: path to data directory
        transform: which of the image transformations to use
        photos_list: list of all paths to photos in the phase
        phase: one of "train", "val", "test"
        labels_dict: dict containing a mapping from category name to labe
    
    """
    
    phase_set = defaultdict(list)
    
    # loop through photos and create mapping from photo to sketch
    total_sketches = 0
    for photo_path in photos_list:
        num_sketches_for_photo = 0
        sketch_dir = prefix + "sketch/" + transform
        photo_dir = prefix + "photo/" + transform
        category, photo_name = photo_path.split('/')
        
        for sketch_name in os.listdir(os.path.join(sketch_dir, category)):
            if sketch_name.split('-')[0] == photo_name:
                phase_set['Photo Path'].append(os.path.join(photo_dir, photo_path) + ".jpg")
                phase_set['Sketch Path'].append(os.path.join(sketch_dir, category, sketch_name))
                num_sketches_for_photo += 1
        
        if num_sketches_for_photo < 5:
            print(photo_path)
            print("Less than five photos for {}".format(photo_path))
            print("Only {}".format(num_sketches_for_photo))
        
        total_sketches += num_sketches_for_photo
    
    print("Total number of sketches: {}".format(total_sketches))
    
    phase_set = pd.DataFrame(phase_set)
    
    phase_set['Label'] = phase_set['Photo Path'].apply(lambda x: labels_dict[x.split('/')[-2]])
    
    phase_set.to_csv(os.path.join(prefix, '{}set.csv'.format(phase)))
    
    return phase_set

def get_photo_lists(prefix, transform):
    with open(prefix + "testset.txt", 'r') as f:
        test_photos = f.readlines()
        test_photos = list(map(lambda x: x.split('.')[0], test_photos))
            
    ## Construct validation photos list
    datadir = prefix + "photo/" + transform 
    val_photos = []
    train_photos = []
    
    for photodir in os.listdir(datadir):    
        photos_in_cat = os.listdir(datadir + photodir)
        
        # construct val photos list
        photos_in_cat = [x for x in photos_in_cat if os.path.join(photodir, x.split('.')[0]) not in test_photos]
        cat_photos = np.random.choice(photos_in_cat, size=10, replace=False)
        cat_photos = list(map(lambda x: photodir + "/" + x[:-4], cat_photos))
        val_photos += cat_photos

        # construct train photos list
        cat_photos = [x for x in photos_in_cat if os.path.join(photodir, x.split('.')[0]) not in test_photos and os.path.join(photodir, x.split('.')[0]) not in val_photos]
        cat_photos = list(map(lambda x: photodir + "/" + x[:-4], cat_photos))
        train_photos += cat_photos
        
    return train_photos, val_photos, test_photos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action="store_true", default=False,
                        help='if on local branch, change paths')
    parser.add_argument('--transform', type=str, default="tx_000100000000/",
                        help='the transformation of the iamge to use')
    
    np.random.seed(42)
    
    args = parser.parse_args()
    
    prefix = "/Users/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/" \
                if args.local else "/home/robincheong/data/sketchy/"

    FILE_NAMES=["invalid-ambiguous.txt", "invalid-error.txt", "invalid-pose.txt"]
    
    train_photos, val_photos, test_photos = get_photo_lists(prefix, args.transform)
    
    labels_dict = make_labels_dict(test_photos)
    
    ## Sanity check that there's no overlap
    assert len(set(test_photos) & set(train_photos)) == 0, "TRAIN AND TEST HAVE OVERLAP"
    assert len(set(val_photos) & set(train_photos)) == 0, "TRAIN AND VAL HAVE OVERLAP"
    assert len(set(test_photos) & set(val_photos)) == 0, "VAL AND TEST HAVE OVERLAP"
    
    train_set = create_sketchy_set(prefix, "tx_000100000000/", train_photos, 'train', labels_dict)
    val_set = create_sketchy_set(prefix, "tx_000100000000/", val_photos, 'val', labels_dict)
    test_set = create_sketchy_set(prefix, "tx_000100000000/", test_photos, 'test', labels_dict)
    
    
    
    