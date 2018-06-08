from loaders.EitzDataLoader import EitzDataLoader
from loaders.SketchyDataLoader import SketchyDataLoader
from loaders.datasets.constants import *
from PIL import Image
import random
import numpy as np
import torch.nn as nn
import argparse
import torch
import torchvision.transforms as T
import torchvision.utils as tvutils
from utils import log_metrics
from models.models import SqueezeNet, ResNet, ConvDualVAE, ConvDualAE, ConvSingleAE, ConvSingleVAE, EmbedGAN

def get_loss_fn(dataset, loss_type):
    ce_loss = nn.CrossEntropyLoss()
    
    def mse_loss(input_, target):
        return (input_ - target) ** 2 / len(input_)
    
    if dataset == "eitz" or loss_type == "classify":
        def classify_loss(sketch_logits, photo_logits, labels):
            return ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)

        return classify_loss
    elif dataset == "sketchy":
        
        if loss_type == "binary":
            def binary_loss(sketch_embed, correct_photo_embed, 
                            sketch_logits, photo_logits, labels):
                
                embedding_loss = torch.sum(mse_loss(sketch_embed, correct_photo_embed))
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 
                
            return binary_loss

        elif loss_type == "trip":
            def trip_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, 
                          sketch_logits, photo_logits, labels, alpha=.2):
                d_anchor_positive = torch.sum(mse_loss(sketch_embed, correct_photo_embed), dim=1)
                d_anchor_negative = torch.sum(mse_loss(sketch_embed, same_cat_diff_photo_embed), dim=1)
#                 alpha = torch.sum(d_anchor_positive - d_anchor_negative) / len(d_anchor_positive)
#                 alpha = alpha.detach()
#                 print("alpha: ", alpha)
                embedding_loss = torch.sum(torch.clamp(d_anchor_positive - d_anchor_negative + alpha, min=0))
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 
            
            return trip_loss

        elif loss_type == "quad":
            def quad_loss(sketch_embed, correct_photo_embed, 
                          same_cat_diff_photo_embed, diff_cat_photo_embed, 
                          sketch_logits, photo_logits, labels, alpha=.2):
                
                loss1 = torch.clamp(torch.sum(mse_loss(sketch_embed, correct_photo_embed), dim=1) 
                                  - torch.sum(mse_loss(sketch_embed, same_cat_diff_photo_embed), dim=1) + alpha, min=0)
                loss2 = torch.clamp(torch.sum(mse_loss(sketch_embed, correct_photo_embed), dim=1)
                                  - torch.sum(mse_loss(sketch_embed, diff_cat_photo_embed), dim=1) + alpha, min=0)
                loss3 = torch.clamp(torch.sum(mse_loss(sketch_embed, same_cat_diff_photo_embed), dim=1)
                                  - torch.sum(mse_loss(sketch_embed, diff_cat_photo_embed), dim=1) + alpha, min=0)
                embedding_loss = torch.sum(loss1 + loss2 + loss3)
                classification_loss = ce_loss(sketch_logits, labels) + ce_loss(photo_logits, labels)
                return embedding_loss, classification_loss 

            return quad_loss
        
        elif loss_type in ["vae", "vae+embed", 'vae+embed+classify']:
            # CITE https://github.com/3ammor/Variational-Autoencoder-pytorch/blob/master/graph/mse_loss.py
            def vae_loss(recon_x, x, mu, logvar):
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return kl_divergence, torch.sum(mse_loss(recon_x, x))
            
            return vae_loss
        
        elif loss_type in ['ae', 'ae+embed', 'ae+embed+classify']:
            def ae_loss(recon_x, x): 
                return torch.sum(mse_loss(recon_x, x))
            
            return ae_loss
        
        elif loss_type in ['gan']:
            # CITE 231N
            def bce_loss(input, target):
                """
                Numerically stable version of the binary cross-entropy loss function.
                As per https://github.com/pytorch/pytorch/issues/751
                See the TensorFlow docs for a derivation of this formula:
                https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
                Inputs:
                - input: PyTorch Tensor of shape (N, ) giving scores.
                - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
                Returns:
                - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
                """
                neg_abs = - input.abs()
                loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
                return loss.mean()

    
    
            def gan_loss(logits_real, logits_fake, device):
                N = logits_real.shape[0]
                d_loss = bce_loss(logits_real, torch.ones(N).to(device)) + bce_loss(logits_fake, torch.zeros(N).to(device))
                g_loss = bce_loss(logits_fake, torch.ones(N).to(device))
                return d_loss, g_loss
            
            return gan_loss
        
    raise ValueError
    

def load_model(args, device):
    if args.model == "resnet":
        model = ResNet()
    elif args.model == "squeezenet":
        model = SqueezeNet(args)
    elif args.model == "ConvDualVAE":
        if args.ftr_extractor_path:
            ftr_extractor = ResNet()
            ftr_extractor.load_state_dict(torch.load(args.ftr_extractor_path))
        model = ConvDualVAE(args, ftr_extractor)
    elif args.model == "ConvDualAE":
        if args.ftr_extractor_path:
            ftr_extractor = ResNet()
            ftr_extractor.load_state_dict(torch.load(args.ftr_extractor_path))
        model = ConvDualAE(args, ftr_extractor)
    elif args.model == "ConvSingleVAE":
        if args.ftr_extractor_path:
            ftr_extractor = ResNet()
            ftr_extractor.load_state_dict(torch.load(args.ftr_extractor_path))
        model = ConvSingleVAE(args, ftr_extractor)
    elif args.model == "ConvSingleAE":
        if args.ftr_extractor_path:
            ftr_extractor = ResNet()
            ftr_extractor.load_state_dict(torch.load(args.ftr_extractor_path))
        model = ConvSingleAE(args, ftr_extractor)
    elif args.model == 'EmbedGAN':
        model = EmbedGAN(args)
    else:
        raise ValueError
        
    model.to(device)
        
    if args.ckpt_path:
        print('loading model')
        model.load_state_dict(torch.load(args.ckpt_path))
    
    return model



def vae_forward(inputs, labels, model, loss_fn, writer, 
                device, batch_num, alpha, N, name, modality,
                compare_embed=False,
                classify=False):
    metrics = {}
    if modality in ['both', 'sketch']:
        sketches = torch.index_select(inputs, 0, torch.tensor(range(0, N)).to(device))
        sketch_embed, recon_sketch, sketch_mu, sketch_logvar = model.forward(sketches, is_sketch=True)

    if modality in ['both', 'photo']:
        photos = torch.index_select(inputs, 0, torch.tensor(range(N, 2 * N)).to(device))
        photo_embed, recon_photo, photo_mu, photo_logvar = model.forward(photos, is_sketch=False)

    if batch_num >= 500 and batch_num % 500 == 0:
        if modality in ['both', 'sketch']:
            tvutils.save_image(recon_sketch, 
                               '/home/robincheong/sketch2img/generated/{}_recon_sketch_{}.png'.format(name, batch_num))
            tvutils.save_image(sketches, 
                               '/home/robincheong/sketch2img/generated/{}_sketches_{}.png'.format(name, batch_num))
        if modality in ['both', 'photo']:
            tvutils.save_image(recon_photo, 
                               '/home/robincheong/sketch2img/generated/{}_recon_photo_{}.png'.format(name, batch_num))
            tvutils.save_image(photos, 
                               '/home/robincheong/sketch2img/generated/{}_photos_{}.png'.format(name, batch_num))
    if modality in ['both', 'sketch']:    
        metrics['sketch_kl_divergence'], metrics['sketch_recon_loss_'] = loss_fn(recon_sketch, sketches, sketch_mu, sketch_logvar)
        metrics['sketch_loss'] = (metrics['sketch_kl_divergence'] + metrics['sketch_recon_loss_']) * alpha
        
    if modality in ['both', 'photo']:
        metrics['photo_kl_divergence'], metrics['photo_recon_loss_'] = loss_fn(recon_photo, photos, photo_mu, photo_logvar)
        metrics['photo_loss'] = (metrics['photo_kl_divergence'] + metrics['photo_recon_loss_']) * alpha
    
    if compare_embed and modality in ['both']:
        metrics['embed_loss'] = torch.sum((sketch_embed - photo_embed) ** 2 / len(sketch_embed))
    else:
        metrics['embed_loss'] = 0

    if classify:
        ce_loss = nn.CrossEntropyLoss()
        if modality == 'photo':
            metrics['classify_loss'] = ce_loss(model.make_predictions(photo_embed), labels) * 10
        elif modality == 'sketch':
            metrics['classify_loss'] = ce_loss(model.make_predictions(sketch_embed), labels) * 10
        else: 
            metrics['classify_loss'] = (ce_loss(model.make_predictions(photo_embed), labels) \
                                       + ce_loss(model.make_predictions(sketch_embed), labels)) * 10
    else:
        metrics['classify_loss'] = 0

    metrics['loss'] = 0
    for metric_name in metrics:
        if metric_name.endswith('loss') and metric_name != 'loss':
            metrics['loss'] += metrics[metric_name] 

    log_metrics(metrics, writer, "batch", batch_num)
    
    return metrics


def ae_forward(inputs, labels, model, loss_fn, writer, device, 
               batch_num, alpha, N, name, modality,
               compare_embed=False,
               classify=False):
    
    metrics = {}
    if modality in ['both', 'sketch']:
        sketches = torch.index_select(inputs, 0, torch.tensor(range(0, N)).to(device))
        sketch_embed, recon_sketch = model.forward(sketches, is_sketch=True)

    if modality in ['both', 'photo']:
        photos = torch.index_select(inputs, 0, torch.tensor(range(N, 2 * N)).to(device))
        photo_embed, recon_photo = model.forward(photos, is_sketch=False)

    if batch_num >= 500 and batch_num % 500 == 0:
        if modality in ['both', 'sketch']:
            tvutils.save_image(recon_sketch, 
                               '/home/robincheong/sketch2img/generated/{}_recon_sketch_{}.png'.format(name, batch_num))
            tvutils.save_image(sketches, 
                               '/home/robincheong/sketch2img/generated/{}_sketches_{}.png'.format(name, batch_num))
        if modality in ['both', 'photo']:
            tvutils.save_image(recon_photo, 
                               '/home/robincheong/sketch2img/generated/{}_recon_photo_{}.png'.format(name, batch_num))
            tvutils.save_image(photos, 
                               '/home/robincheong/sketch2img/generated/{}_photos_{}.png'.format(name, batch_num))

    if modality in ['both', 'sketch']:    
        metrics['sketch_recon_loss'] = loss_fn(recon_sketch, sketches) * alpha
    if modality in ['both', 'photo']:
        metrics['photo_recon_loss'] = loss_fn(recon_photo, photos) * alpha

    if compare_embed and modality in ['both']:
        metrics['embed_loss'] = torch.sum((sketch_embed - photo_embed) ** 2 / len(sketch_embed))
    else:
        metrics['embed_loss'] = 0
        
    if classify:
        ce_loss = nn.CrossEntropyLoss()
        if modality == 'photo':
            metrics['classify_loss'] = ce_loss(model.make_predictions(photo_embed), labels) * 10
        elif modality == 'sketch':
            metrics['classify_loss'] = ce_loss(model.make_predictions(sketch_embed), labels) * 10
        else: 
            metrics['classify_loss'] = (ce_loss(model.make_predictions(photo_embed), labels) \
                                       + ce_loss(model.make_predictions(sketch_embed), labels)) * 10
    else:
        metrics['classify_loss'] = 0
    
    metrics['loss'] = 0
    for metric_name in metrics:
        if metric_name.endswith('loss') and metric_name != 'loss':
            metrics['loss'] += metrics[metric_name] 
    
    log_metrics(metrics, writer, "batch", batch_num)
    
    return metrics


def classify_contrast_forward(inputs, labels, model, loss_fn, writer, device, 
                              batch_num, alpha, loss_type, N):
    metrics = {}
    features = model.extract_features(inputs)
    indices = torch.tensor(range(0, 2 * N)).to(device)
    selected_features = torch.index_select(features, 0, indices)
    logits = model.make_predictions(selected_features)
    sketch_logits, photo_logits = torch.split(logits, N)

    if loss_type == "classify":
        metrics['loss'] = loss_fn(sketch_logits, photo_logits, labels)
    else:
        # reorganize into photo embeds and sketch embeds
        # feed in embed for photo and sketch
        metrics['embedding_loss'], metrics['classification_loss'] = loss_fn(*torch.split(features, N), 
                                                                                sketch_logits, photo_logits, labels)

        metrics['loss'] = alpha * metrics['embedding_loss'] + (1 - alpha) * metrics['classification_loss']

    _, sketch_preds = torch.max(sketch_logits, 1)
    _, photo_preds = torch.max(photo_logits, 1)

    sketch_cor = sum(sketch_preds.cpu().numpy() == labels.cpu().numpy()) 
    photo_cor = sum(photo_preds.cpu().numpy() == labels.cpu().numpy())

    metrics['sketch_cor'] = sketch_cor
    metrics['photo_cor'] = photo_cor

    log_metrics(metrics, writer, "batch", batch_num)

    # TODO Change this to be args.verbose
    if True:
        print("=" * 100)
        print("Predicted classes for sketches: {}".format(sketch_preds.cpu().tolist()))
        print("Predicted classes for photos: {}".format(photo_preds.cpu().tolist()))
        print("Ground truth: {}".format(labels.cpu().tolist()))
        print("=" * 100)
                    

    return metrics

def gan_forward(inputs, labels, model, loss_fn, writer, device, batch_num, N):
    metrics = {}
    sketches, photos = torch.split(inputs, N)
    logits_real, logits_fake = model(sketches, photos)
    
    
    d_loss, g_loss = loss_fn(logits_real, logits_fake, device)
    
    metrics['d_loss'] = d_loss
    metrics['g_loss'] = g_loss
    
    metrics['loss'] = 0
    for metric_name in metrics:
        if metric_name.endswith('loss') and metric_name != 'loss':
            metrics['loss'] += metrics[metric_name] 
    
    log_metrics(metrics, writer, "batch", batch_num)
    
    return metrics