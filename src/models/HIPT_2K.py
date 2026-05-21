### Dependencies
# Base Dependencies
import os
import pickle
import sys
sys.path.append(os.path.abspath('/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src')) 

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
from torchvision import transforms
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.hipt_utils import get_vit256, get_vit2k

print("device:", torch.cuda.get_device_name())

print(torch.__version__)

class HIPT_2K(torch.nn.Module):
    """
    HIPT Model (ViT-2K) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
    """
    def __init__(self, 
        model256_path: str = '../Checkpoints/vit256_small_dino.pth',
        model2k_path: str = '../Checkpoints/vit2k_xs_dino.pth', 
        device256=torch.device('cuda:0'), 
        device2k=torch.device('cuda:0')):

        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path, arch='vit_256_custom', device=device256).to(device256)
        self.model2k = get_vit2k(pretrained_weights=model2k_path, arch='vit_2k_custom', device=device2k).to(device2k)
        self.device256 = device256
        self.device2k = device2k
    
    def forward(self, x):
        """
        Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT-2K.
        1. x is center-cropped such that the W / H is divisible by the patch token size in ViT-2K (e.g. - 256 x 256).
        2. x then gets unfolded into a "batch" of [256 x 256] images.
        3. A pretrained ViT-256 model extracts the CLS token from each [256 x 256] image in the batch.
        4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
        5. This feature grid is then used as the input to ViT-2K, outputting [CLS]_2K.
        
        Args:
            - x (torch.Tensor): [1 x C x W' x H'] image tensor.
        
        Return:
            - features_cls2k (torch.Tensor): [1 x 192] cls token (d_2k = 192 by default).
        """
        
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H] 
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] [1 x 3 x 8 x 8 x 256 x 256]
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        features_cls256 = []
        for mini_bs in range(0, batch_256.shape[0], 64):                       # 3. B may be too large for ViT-256. We further take minibatches of 64.
            minibatch_256 = batch_256[mini_bs:mini_bs+64].to(self.device256, non_blocking=True)
            features_cls256.append(self.model256(minibatch_256).detach().cpu()) # 3. Extracting ViT-256 features from [64 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
        features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0) 
        features_cls256 = features_cls256.to(self.device2k, non_blocking=True)  # 4. [1 x 384 x w_256 x h_256]
        features_cls2k = self.model2k.forward(features_cls256)                  # 5. [1 x 192], where 192 == dim of ViT-2K [ClS] token.
        return features_cls2k
    
    def generate_vit256_embeddings(self, x):
        """
        Helper function that generates ViT-256 features from [256 x 256] image tensors.
        
        Args:
            - x (torch.Tensor): [B x 3 x 256 x 256] image tensor batch.
        Returs:
            - features_cls256 (torch.Tensor): [B x 384] ViT-256 CLS token features.
        """
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H] 
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] [1 x 3 x 8 x 8 x 256 x 256]
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        features_cls256 = []
        for mini_bs in range(0, batch_256.shape[0], 64):                       # 3. B may be too large for ViT-256. We further take minibatches of 64.
            minibatch_256 = batch_256[mini_bs:mini_bs+64].to(self.device256, non_blocking=True)
            features_cls256.append(self.model256(minibatch_256).detach().cpu()) # 3. Extracting ViT-256 features from [64 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
        return features_cls256

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        """
        Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
        are divisible by 256.
        
        (Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
        Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
        
        Args:
            - img (torch.Tensor): [1 x C x W' x H'] image tensor.
            - patch_size (int): Desired patch size to evenly subdivide the image.
        
        Return:
            - img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
            - w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
            - h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
        """
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)
        return img_new, w_256, h_256