"""
Reusable modules
"""
import torch
import torch.nn as nn
import math
from models.attention import MHA

class PatchEmbed(nn.Module):
    """
    Image to Patch embedding with flexible dimensions
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=384
    ):
        super().__init__()
        self.img_size=img_size # this is the size of the input image when cropped during DINO augmentation
        self.patch_size=patch_size # this is the size of the patches to get from images, last layer of HIPT.
        self.num_patches=(img_size // patch_size) **2
        self.grid_size=img_size // patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1,2) # reshape to (B, num_patches, embed_dim)
        return x
        

class MLP(nn.Module):
    """
    MLP implementation as in Vision Transformer
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            dropout=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features # hidden feature is the Embedding dimension * MLP Ratio (384*4))
        
        self.fc1=nn.Linear(in_features, hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class TransformerBlock(nn.Module):
    """
    Transformer Block for ViTs
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop_path=0.
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                        attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, dropout=drop)
        
    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class RegionPool():
    """
    Class to pool patch-level features to produce region-level features
    """
    pass

class TokenAggregator():
    """
    Class to aggregate top-k patches dense grid patch tokens into cellular representations
    """
    pass