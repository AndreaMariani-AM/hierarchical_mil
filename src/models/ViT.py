# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Flexible Vision Transformer Implementation for Hierarchical Training
Supports arbitrary image sizes, patch sizes, and dimensions

Reimplementation of ViT with modifications from:
https://github.com/mahmoodlab/HIPT/blob/master/1-Hierarchical-Pretraining/vision_transformer.py
"""
from typing import Callable

import torch
import torch.nn as nn
import math
from functools import partial
import os
import sys
import numpy as np
from PIL import Image
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
sys.path.append(os.path.abspath('/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src')) 
# from models.attention import MHA
from models.modules import PatchEmbed, TransformerBlock
from lazyslide.models.vision import UNI2, Virchow2

class UNI2Dense(UNI2):
    name = "UNI2Dense"
    # UNI2 returns embeddings of size 1536, there is no concat between CLS and patch embeddings
    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor, transform: Callable = None) -> torch.Tensor:
        if transform is not None:
            image = transform(image)
        tokens = self.model.forward_features(image) # returns the full [cls || reg || patch] tokens
        patch_tokens = tokens[:, 1+8:, :] # assuming 1 is CLS and 8 are reg tokens
        return patch_tokens #(1, 256, 1536)

class Virchow2Dense(Virchow2):
    name = "Virchow2Dense"
    # Virchow2 returns embeddings of size 1280
    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor, transform: Callable = None) -> torch.Tensor:
        if transform is not None:
            image = transform(image)
        tokens = self.model(image) #returns the full [cls || reg || patch] tokens
        patch_cls = tokens[:, 0] # (1, 1280)
        patch_tokens = tokens[:, 5:] # patch tokens (256, 1280)
        region_tokens  = torch.cat((patch_cls, patch_tokens.mean(1)), dim=-1)
        return region_tokens, patch_cls, patch_tokens # (1, 256, 1280)


class VisionTransformer(nn.Module):
    """
    Flexible Vision Transformer
    
    Args:
        img_size: Input image size
        patch_size: Size of image patches
        in_channels: Number of input channels (3 for RGB), will be embed_dim for hierarchical setup
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Enable bias for qkv if True
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
    """
    def __init__(
            self,
            img_size=[224], # this is the cropping size
            patch_size=16,
            in_channels=3,
            num_classes=0,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            **kwargs
    ):
        
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, 
            in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Add stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head (optional)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional encodings for different image sizes
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        # Add a small number to avoid floating point error
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        """
        Forwarding method to extract CLS token
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return cls token

    def forward(self, x):
        """
        Send CLS token to the head
        """
        x = self.forward_features(x)
        x = self.head(x) #for pretraining this will ben nn.Identity()
        return x

    def get_last_selfattention(self, x):
        """
        Return attention weights from the last block
        """
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


def vit_tiny(patch_size=16, **kwargs):
    """
    ViT-Tiny: embed_dim=192, depth=12, num_heads=3
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    """
    ViT-Small: embed_dim=384, depth=12, num_heads=6
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    """
    ViT-Base: embed_dim=768, depth=12, num_heads=12
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# Custom configurations for your hierarchical setup
def vit_256_custom(patch_size=32, **kwargs):
    """
    ViT for 256×256 images with 32×32 patches (8×8 = 64 tokens)
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

class VisionTransformer2K(nn.Module):
    """
    Flexible Vision Transformer
    
    Args:
        img_size: Input image size
        patch_size: Size of image patches
        in_chans: Number of input channels (3 for RGB, could be embed_dim for hierarchical)
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Enable bias for qkv if True
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        norm_layer: Normalization layer
    """
    def __init__(
            self,
            img_size=[224],
            patch_size=16,
            input_emb_dim=384,
            output_emb_dim=192,
            num_classes=0,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            **kwargs
    ):
        
        super().__init__()
        embed_dim = output_emb_dim
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.phi = nn.Sequential(*[nn.Linear(input_emb_dim, output_emb_dim), nn.GELU(), nn.Dropout(p=drop_rate)])
        num_patches = int(img_size[0] // 16)**2
        print("# of Patches:", num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Add stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) 
                for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head (optional)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional encodings for different image sizes
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        
        w0 = w // 1
        h0 = h // 1
        
        # Add a small number to avoid floating point error
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x):
        """
        Forwarding method to extract CLS token
        """
        self.mpp_feature = x
        B, embed_dim, H, W = x.shape
        x = x.flatten(2,3).transpose(1,2)

        x = self.phi(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return cls token

    def forward(self, x):
        """
        Send CLS token to the head
        """
        x = self.forward_features(x)
        x = self.head(x) #for pretraining this will ben nn.Identity()
        return x

    def get_last_selfattention(self, x):
        """
        Return attention weights from the last block
        """
        self.mpp_feature = x
        B, embed_dim, H, W = x.shape
        x = x.flatten(2,3).transpose(1,2)

        x = self.phi(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


def vit_2k_custom(patch_size=256, in_chans=384, **kwargs):
    """
    ViT for 2048×2048 images with 256×256 patches (8×8 = 64 patches)
    in_chans=384 to accept features from previous ViT stage
    """
    model = VisionTransformer2K(
        patch_size=patch_size, input_emb_dim=in_chans,
        output_emb_dim=192, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DINOHead(nn.Module):
    """
    DINO projection head for self-supervised learning
    
    Args:    
        in_dim: Input dimension (embedding dimension from ViT)
        out_dim: Output dimension (projection dimension for DINO loss)
        use_bn: Whether to use batch normalization in the MLP
        norm_last_layer: Whether to normalize the last layer's weights (as in DINO)
        nlayers: Number of layers in the MLP (including the output layer)
        hidden_dim: Hidden dimension for the MLP layers
        bottleneck_dim: Dimension of the bottleneck layer before the final projection
    """
    def __init__(
            self,
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256
    ):
        
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x