import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# Attention class

class AttnVanilla(nn.Module):
    """
    Standard attention mechanism
    """
    def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 128
    ):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        attn_score = self.attn(x)
        return attn_score

class GatedAttn(nn.Module):
    """
    Gated Attention
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_classes: int = 2
    ):
        super().__init__()

        self.attn_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        ) # representations as in vanilla attention

        self.attn_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        ) # this are "gates" values [0,1]

        self.attn_w = nn.Linear(hidden_dim, n_classes) # controls the flow of which representations pass the gate

    def forward(
            self,
            x: torch.Tensor
    ):
        A_V = self.attn_V(x)
        A_U = self.attn_U(x)
        A = self.attn_w(A_V.mul(A_U))

        return A

class MHA(nn.Module):
    """
    Multi Head self Attention for Vits
    """
    def __init__(
        self,
        dim,
        num_heads = 8,
        qkv_bias = False,
        attn_drop=0.,
        proj_drop=0.    
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop =nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # (batch, n_tokens, embeddings_dim)
        # from (batch, n_tokens, emb_dim*3) --> (batch, n_tokens, [q,k,v], n_heads, head_dim) --> (qkv, batch, num_heads, n_tokens, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4 )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # q is (B, heads, tokens, head_dim), k is (B, heads, head_dim, tokens) --> (B, heads, token, token)
        attn = attn.softmax(dim=-1) # or attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # attn@ v is (B, heads, tokens, head_dim), transpose get me (B, tokens, heads, head_dim) --> reshape to (B, tokens, embedding_dim) embeddng_dim = heads*head_dim
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
        