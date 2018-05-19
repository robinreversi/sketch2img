from torch.utils.data import DataLoader
from .datasets.EitzDataset import EitzDataset


class EitzDataLoader(DataLoader):
    """Data loader for the Eitz2012 Dataset"""
    def __init__(self, args, phase):
        dataset = EitzDataset(phase, args.img_size)
        is_training = phase == 'train'
        super().__init__(dataset,
                         batch_size=batch_size if is_training else 1,
                         shuffle=is_training,
                         num_workers=num_threads if is_training else 1)