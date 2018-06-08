import numpy as np
EITZ_FP = "/home/robincheong/data/eitz2012/"
PHOTO_DIR = "/home/robincheong/data/sketchy/photo/tx_000100000000/"
SKETCH_DIR = "/home/robincheong/data/sketchy/sketch/tx_000100000000/"

# EITZ_FP = "/home/robincheong/Documents/Stanford/CS231N/Project/data/eitz2012/"
# PHOTO_DIR = "/home/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/photo/tx_000000000000/"
# SKETCH_DIR = "/home/robincheong/Documents/Stanford/CS231N/Project/data/sketchy/sketch/tx_000000000000/"

PHOTO_MEAN = np.array([0.4489105882, 0.4489105882, 0.4489105882], dtype=np.float32)
SKETCH_MEAN = np.array([0.95263444, 0.95263444, 0.95263444], dtype=np.float32)

SKETCH_STD = np.array([0.35874852, 0.35874852, 0.35874852], dtype=np.float32)
PHOTO_STD = np.array([0.4323276436, 0.4323276436, 0.4323276436], dtype=np.float32)

AE_CRITERIA = ['sketch_recon_loss', 'photo_recon_loss', 'embed_loss', 'loss', 'classify_loss']

VAE_CRITERIA = AE_CRITERIA + ['sketch_kl_divergence', 'photo_kl_divergence', 'sketch_loss', 'photo_loss'] 

CLASSIFY_CRITERIA = ['sketch_cor', 'photo_cor', 'loss']

CLASSIFY_CONTRAST_CRITERIA = CLASSIFY_CRITERIA + ['embedding_loss', 'classification_loss']
