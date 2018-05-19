import argparse
import getpass
import json
import os
import torch


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/home/robincheong/sbir/checkpoints/',
                                 help='Directory in which to save checkpoints.')
        self.parser.add_argument('--checkpoint_path', type=str, default='',
                                 help='Path to checkpoint to load. If empty, start from scratch.')
        self.parser.add_argument('--data_dir', type=str,
                                 help='Path to data directory')
        self.parser.add_argument('--name', type=str, help='Experiment name.')
        self.parser.add_argument('--img_format', type=str, default='png', choices=('jpg', 'png'),
                                 help='Format for input images')
        self.parser.add_argument('--resize_shape', type=str, default='-1',
                                 help='Comma-separated 2D shape for images after resizing (before cropping).')
        self.parser.add_argument('--crop_shape', type=str, default='448,448',
                                 help='Comma-separated 2D shape for images after cropping (crop comes after resize).')
        self.parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--toy', action='store_true', default=0, help='Use small dataset or not.')
        self.parser.add_argument('--toy_size', type=int, default=5,
                                 help='How many of each type to include in the toy dataset.')

    def parse_args(self):
        args = self.parser.parse_args()

        # Set up save dir and output dir (test mode only)
        args.save_dir = os.path.join(args.checkpoints_dir, '{}_{}'.format(getpass.getuser(), args.name))
        os.makedirs(args.save_dir, exist_ok=True)
        if not self.is_training:
            args.results_dir = os.path.join(args.results_dir, '{}'.format(getpass.getuser()))
            os.makedirs(args.results_dir, exist_ok=True)

        # Save args to a JSON file
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        return args
