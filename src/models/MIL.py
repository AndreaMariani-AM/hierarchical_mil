import os
from pathlib import Path
import sys
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
sys.path.append(os.path.abspath('/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src')) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal, Dict
# from src.models.attention import AttnVanilla, GatedAttn
from models.attention import AttnVanilla, GatedAttn
from models.experts_MIL import HierarchicalMIL # re-exported for easier imports in trainer.py
import lightning as L

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# Main MIL model class
class AttentionMIL(nn.Module):
    """
    Simple 2 Layer attention based MIL model.
    Finds important patches via attention and then classify.
    """
    def __init__(
            self,
            input_dim: int = 2560,
            hidden_dim: int = 1280,
            hidden_dim_2: int = 256,
            n_classes: int = 2,
            dropout: float = 0.25,
            instance_batch_size: int = 512
    ):
        """
        Args:
            instance_batch_size: Process instances in batches due to memory constrainsts
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.instance_batch_size = instance_batch_size

        # instance-level features projection
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # (1, n_instances, n_features) --> (1, n_instances, hidden_dim)
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim_2) # (1, n_instances, hidden_dim) --> (1, n_instances, hidden_dim_2)
        )

        # self.attn = AttnVanilla(hidden_dim_2, hidden_dim_2 // 2)
        self.attn = GatedAttn(hidden_dim_2, hidden_dim_2 // 2, n_classes=n_classes)

        #Bag-level classifier
        self.cls = nn.Linear(hidden_dim_2, 1) # (hidden_dim_2,) --> (1,) for binary classification

    def forward(
            self,
            x: torch.Tensor,
            attn_return: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            x: Tensor of shape (1, n_instances, n_features), a single bag of instances is a WSI
            attn_return: Whether to return attention weights
        Returns:
            logits: Tensor of shape (n_classes,)
            attn_weights: Optional tensor of shape (n_instances, n_classes)
            contributions: Always None for AttentionMIL (no per-instance predictions)
        """
        x = x.squeeze(0) #remove dataloader batch dimension
        n_instances = x.shape[0]

        # Process instances in batches
        all_h = []
        all_attn_score = []

        for i in range(0, n_instances, self.instance_batch_size):
            batch = x[i:i + self.instance_batch_size]

            #run features through the network
            h = self.instance_encoder(batch) # (batch, n_featues) --> (batch, hidden_dim_2)
            all_h.append(h)

            # Compute attn
            attn_score = self.attn(h)  # (batch, hidden_dim_2) --> (batch, n_classes) for gated attention
            all_attn_score.append(attn_score)

        # Concatenate batches
        h = torch.cat(all_h, dim=0) # (n_instances, hidden_dim_2)
        attn_score = torch.cat(all_attn_score, dim=0) # (n_instances, n_classes)

        # Compute attention weights across all instances
        attn_weights = F.softmax(attn_score, dim=0)
        
        # Aggregate using attention
        bag_representation = torch.sum(attn_weights * h, dim=0) # (hidden_dim_2,)

        # Use bag representations to classify 
        logits = self.cls(bag_representation) # (n_classes,)

        if attn_return:
            return logits, attn_weights, None
        return logits, None, None


class AdditiveMIL(nn.Module):
    """
    Hybrid Additive Attention MIL model.
    Combines per-class gated attention with per-instance classification.
    Each instance gets per-class attention scores and per-class predictions;
    the bag logit for each class is the aggregation of
    attention_weight * instance_prediction across all instances.
    """
    def __init__(
            self,
            input_dim: int = 1280,
            hidden_dim: int = 512,
            hidden_dim_2: int = 256,
            n_classes: int = 1,
            dropout: float = 0.25,
            instance_batch_size: int = 512,
            pooling: Literal["sum", "mean", "max"] = "sum"
    ):
        """
        Args:
            input_dim: Dimension of input instance features (e.g. 1280 for Virchow2)
            hidden_dim: First hidden layer dimension
            hidden_dim_2: Second hidden layer dimension (instance representation dim)
            n_classes: Number of output classes. Use 1 for binary (BCE loss),
                       >1 for multi-class (CrossEntropy loss)
            dropout: Dropout rate
            instance_batch_size: Process instances in batches for memory efficiency
            pooling: Aggregation strategy - "sum" (default), "mean", or "max"
        """
        super().__init__()
        self.n_classes = n_classes
        self.instance_batch_size = instance_batch_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        # Instance-level feature projection (shared across all classes)
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim_2)
        )

        # Per-class gated attention: produces (n_instances, n_classes) attention scores
        self.attn = GatedAttn(hidden_dim_2, hidden_dim_2 // 2, n_classes=n_classes)

        # Per-class instance-level classifiers
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim_2, 1) for _ in range(n_classes)]
        )

        # Pooling dispatch
        self._pool_fn: Dict[str, callable] = {
            "sum": lambda x, dim: torch.sum(x, dim=dim),
            "mean": lambda x, dim: torch.mean(x, dim=dim),
            "max": lambda x, dim: torch.max(x, dim=dim).values,
        }
        
        # Init weights
        self.apply(initialize_weights)

    def forward(
            self,
            x: torch.Tensor,
            attn_return: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (1, n_instances, n_features), a single bag (WSI)
            attn_return: Whether to return attention weights and contributions
        Returns:
            logits: Tensor of shape (n_classes,)
            attn_weights: Optional tensor of shape (n_instances, n_classes)
            contributions: Optional tensor of shape (n_instances, n_classes),
                           per-instance, per-class contribution scores
        """
        x = x.squeeze(0)  # remove dataloader batch dimension
        n_instances = x.shape[0]

        # Process instances in batches for memory efficiency
        all_h = []
        all_attn_score = []

        for i in range(0, n_instances, self.instance_batch_size):
            batch = x[i:i + self.instance_batch_size]

            # Encode instances
            h = self.instance_encoder(batch)  # (batch, hidden_dim_2)
            all_h.append(h)

            # Per-class attention scores
            attn_score = self.attn(h)  # (batch, n_classes)
            all_attn_score.append(attn_score)

        # Concatenate all batches
        h = torch.cat(all_h, dim=0)  # (n_instances, hidden_dim_2)
        attn_score = torch.cat(all_attn_score, dim=0)  # (n_instances, n_classes)

        # Softmax attention weights across instances for each class
        attn_weights = F.softmax(attn_score, dim=0)  # (n_instances, n_classes)

        # Per-class instance predictions: for each class c, classify every instance
        # contributions[i, c] = attn_weights[i, c] * classifier_c(h_i)
        contributions = torch.zeros(n_instances, self.n_classes, device=h.device)
        for c in range(self.n_classes):
            instance_preds = self.instance_classifiers[c](h).squeeze(-1)  # (n_instances,)
            contributions[:, c] = attn_weights[:, c] * instance_preds

        # Aggregate contributions across instances
        pool_fn = self._pool_fn[self.pooling]
        logits = pool_fn(contributions, dim=0)  # (n_classes,)
        logits = torch.clamp(logits, min=-7, max=7)

        if attn_return:
            return logits, attn_weights, contributions
        return logits, None, None



