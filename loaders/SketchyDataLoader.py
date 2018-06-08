from torch.utils.data import DataLoader
from .datasets.SketchyDataset import SketchyDataset


class SketchyDataLoader(DataLoader):
    """Data loader for the Sketchy Dataset"""
    def __init__(self, args, phase):
        dataset = SketchyDataset(args, phase)
        super().__init__(dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_threads)
