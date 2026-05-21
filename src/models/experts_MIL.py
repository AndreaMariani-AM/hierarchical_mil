"""
Hierarchical Expert-based Multiple Instance Learning (MIL).

Three specialised experts look at different spatial scales of a WSI:

    1. **PatchExpert**  – Additive MIL on 256×256 patch embeddings.
       Produces per-patch contributions and exposes the *top-k attended*
       patch indices so the CellExpert can zoom in.

    2. **RegionExpert** – For every 2048×2048 region, builds a region
       representation as ``MLP(concat(gated_attn_vector, mean_patch_embds))``
       then applies additive MIL across regions.

    3. **CellExpert**  – Takes the top-k 256×256 patches selected by
       PatchExpert, extracts a dense 16×16 grid of sub-patch token
       embeddings via a *frozen* foundation model (UNI2 / Virchow2 via
       lazyslide), and runs additive MIL on those cell-level tokens.

All three experts return the same ``ExpertOutput`` named-tuple so
``HierarchicalMIL`` can concatenate their representations and
predict the slide label via a fusion MLP + learnable expert alphas.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import make_dataclass

# ── project imports ────────────────────────────────────────────────────
sys.path.append(
    os.path.abspath(
        "/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src"
    )
)
from models.attention import GatedAttn

# ======================================================================
# 0.  Shared helpers & data-structures
# ======================================================================

def initialize_weights(module: nn.Module) -> None:
    """Xavier-normal for Linear layers, unit-scale for LayerNorm.

    Designed to be used with ``nn.Module.apply`` which already recurses
    through all submodules, so this function only inspects *module*
    itself (no inner ``module.modules()`` loop).
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


@dataclass
class ExpertOutput:
    """
    Standardised output returned by every expert.

    Attributes
    ----------
    logits : Tensor, shape ``(n_classes,)``
        Bag-level logits produced by this expert (before fusion).
    attn_weights : Tensor, shape ``(n_instances, n_classes)``
        Softmax-normalised attention weights for each instance × class.
    contributions : Tensor, shape ``(n_instances, n_classes)``
        Per-instance, per-class contribution scores
        (``attn_weight * instance_prediction``).
    representation : Tensor, shape ``(repr_dim,)``
        A compact expert-level representation vector that is
        concatenated across experts in the fusion head.
    """
    logits: torch.Tensor
    attn_weights: torch.Tensor
    contributions: torch.Tensor
    representation: torch.Tensor

RegionExpertOutput = make_dataclass(
    "RegionExpertOutput",
    fields=[("region_embeddings", torch.Tensor)],
    bases=(ExpertOutput,)
    )

# ── Pooling dispatch (re-used by every expert) ────────────────────────
_POOL_FNS: Dict[str, callable] = {
    "sum":  lambda x, dim: torch.sum(x, dim=dim),
    "mean": lambda x, dim: torch.mean(x, dim=dim),
    "max":  lambda x, dim: torch.max(x, dim=dim).values,
}


# ======================================================================
# 1.  ClassConditionalAdditiveMIL  – reusable additive-MIL core
# ======================================================================

class ClassConditionalAdditiveMIL(nn.Module):
    """
    Re-usable *additive* MIL block.

    Given a set of instance embeddings ``h`` of shape ``(N, D)``:
        1. Per-class gated attention  → ``(N, C)``
        2. Per-class instance classifiers → ``(N, C)``
        3. Contributions = attn_weight × instance_pred → ``(N, C)``
        4. Pool across instances → logits ``(C,)``

    Also computes an *attention-weighted* bag representation
    ``(D,)`` that can be handed to a downstream fusion head.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the incoming instance embeddings.
    num_classes : int
        Number of target classes (1 for binary BCE, >1 for CE).
    key_dim : int
        Hidden dimension used inside the gated attention module.
    dropout : float
        Dropout applied inside the gated attention gate.
    pooling : str
        How to aggregate per-instance contributions: ``"sum"``
        (default), ``"mean"`` or ``"max"``.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_classes: int = 1,
        key_dim: int = 128,
        dropout: float = 0.0,
        pooling: Literal["sum", "mean", "max"] = "sum",
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling = pooling
        self.chunk_size = chunk_size

        # ── Per-class gated attention ─────────────────────────────────
        # GatedAttn:  input_dim → tanh  +  input_dim → sigmoid  → n_classes
        self.attn = GatedAttn(
            input_dim=in_dim,
            hidden_dim=key_dim,
            n_classes=num_classes,
        )

        # ── Per-class instance-level classifiers ──────────────────────
        # Each classifier maps an instance embedding → 1 scalar per class
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(in_dim, 1) for _ in range(num_classes)]
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : Tensor, shape ``(N, D)``
            Instance embeddings (already projected to expert dim).
        mask : Tensor, shape ``(N,)`` bool, optional
            ``True`` for *valid* instances.  Masked positions get
            ``-inf`` attention before softmax so they contribute zero.

        Returns
        -------
        logits : Tensor ``(C,)``
        attn_weights : Tensor ``(N, C)``
        contributions : Tensor ``(N, C)``
        bag_repr : Tensor ``(D,)``
            Attention-weighted mean of instance embeddings (single
            vector summarising the bag for this expert).
        """
        N, D = h.shape
        all_attn_score = []

        # 1) Per-class raw attention scores ───────────────────────────
        for start in range(0, N, self.chunk_size):
            batch = h[start:start + self.chunk_size]  # (chunk, D)
            all_attn_score.append(self.attn(batch))  # (chunk, C)

        attn_score = torch.cat(all_attn_score, dim=0)  # (N, C)

        # 1a) Optionally mask out padded / invalid instances
        if mask is not None:
            # mask is (N,) bool, True = keep.  Set invalid → -inf
            attn_score = attn_score.masked_fill(
                ~mask.unsqueeze(-1), float("-inf")
            )

        # 2) Softmax across instances (dim=0) for each class ─────────
        attn_weights = F.softmax(attn_score, dim=0)  # (N, C)

        # 3) Per-class instance predictions ───────────────────────────
        #    contributions[i, c] = attn_weights[i, c] * cls_c(h_i)
        contributions = torch.zeros(N, self.num_classes, device=h.device)
        for c in range(self.num_classes):
            inst_pred = self.instance_classifiers[c](h).squeeze(-1)  # (N,)
            contributions[:, c] = attn_weights[:, c] * inst_pred

        # 4) Pool contributions across instances → bag logits ─────────
        pool_fn = _POOL_FNS[self.pooling]
        logits = pool_fn(contributions, dim=0)  # (C,)

        # 5) Attention-weighted bag representation (for fusion) ───────
        #    Use mean across classes for the weighting vector
        avg_attn = attn_weights.mean(dim=1)  # (N,)  – class-averaged
        bag_repr = (avg_attn.unsqueeze(-1) * h).sum(dim=0)  # (D,)

        return logits, attn_weights, contributions, bag_repr


# ======================================================================
# 2.  PatchExpert
# ======================================================================

class PatchExpert(nn.Module):
    """
    MIL expert operating on 256×256 **patch** embeddings.

    Flow
    ----
    1. Project raw foundation-model embeddings (e.g. 2560-d Virchow2)
       down to ``hidden_dim`` via a shared encoder MLP.
    2. Run ``ClassConditionalAdditiveMIL`` to obtain per-patch
       attention, contributions, logits, and a bag representation.
    3. Expose the **top-k** attended patch indices so that
       ``CellExpert`` can select which patches to zoom into.

    Parameters
    ----------
    in_dim : int
        Raw patch embedding dimension (2560 for Virchow2, 1024 for UNI2).
    hidden_dim : int
        Intermediate projection dimension.
    proj_dim : int
        Final instance representation dimension fed to additive MIL.
    num_classes : int
        Number of target classes.
    dropout : float
        Dropout rate in the encoder MLP.
    pooling : str
        Aggregation strategy for additive MIL (``"sum"``, ``"mean"``,
        ``"max"``).
    top_k : int
        Number of highest-attention patches to expose for CellExpert.
    """

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 1280,
        proj_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.25,
        pooling: Literal["sum", "mean", "max"] = "sum",
        top_k: int = 512,
    ):
        super().__init__()
        self.top_k = top_k
        self.proj_dim = proj_dim

        # ── Shared instance encoder: raw_dim → proj_dim ──────────────
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

        # ── Core additive MIL on projected patch embeddings ──────────
        self.mil = ClassConditionalAdditiveMIL(
            in_dim=proj_dim,
            num_classes=num_classes,
            key_dim=proj_dim // 2,
            dropout=dropout,
            pooling=pooling,
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ExpertOutput, torch.Tensor]:
        """
        Parameters
        ----------
        h_patches : Tensor, shape ``(N_patches, D_raw)``
            Raw foundation-model embeddings for every 256×256 patch
            in a single WSI.  Already squeezed (no batch dim).
        mask : Tensor, shape ``(N_patches,)`` bool, optional
            ``True`` = valid patch.

        Returns
        -------
        expert_out : ExpertOutput
            Standardised expert output (logits, attn, contributions,
            bag representation).
        top_k_indices : Tensor, shape ``(K,)`` int64
            Indices (into the *original* patch array) of the top-k
            most-attended patches.  These are passed to CellExpert.
        """
        # 1) Project patches to hidden space ──────────────────────────
        h = self.encoder(h_patches)  # (N, proj_dim)

        # 2) Additive MIL ─────────────────────────────────────────────
        logits, attn_w, contribs, bag_repr = self.mil(h, mask=mask)

        # 3) Select top-k attended patches ────────────────────────────
        #    Average attention across classes to get a single per-patch
        #    importance score, then pick the highest-K indices.
        avg_attn = attn_w.mean(dim=1)  # (N,)
        k = min(self.top_k, avg_attn.shape[0])
        top_k_indices = torch.topk(avg_attn, k=k).indices  # (K,)

        expert_out = ExpertOutput(
            logits=logits,
            attn_weights=attn_w,
            contributions=contribs,
            representation=bag_repr,
        )

        return expert_out, top_k_indices.to('cpu')


# ======================================================================
# 3.  RegionExpert
# ======================================================================

class RegionExpert(nn.Module):
    """
    MIL expert operating at the **region** (2048×2048) level.

    For each region *r* the representation is built as::

        region_repr_r = MLP( concat( gated_attn_vector_r , mean_patch_embds_r ) )

    where:
    - ``gated_attn_vector_r`` is the attention-weighted sum of patch
      embeddings in region *r* (using a **per-class** gated attention
      module internal to this expert).
    - ``mean_patch_embds_r`` is the simple mean of patch embeddings
      in region *r*.

    After computing a region representation for every region, the
    expert runs ``ClassConditionalAdditiveMIL`` across regions.

    Parameters
    ----------
    in_dim : int
        Raw patch embedding dim (e.g. 2560 Virchow2).
    hidden_dim : int
        Intermediate projection size for the patch encoder.
    proj_dim : int
        Dimension of the projected patch embeddings *before* region
        pooling / attention.
    region_repr_dim : int
        Output dim of the region representation MLP.
    num_classes : int
        Number of target classes.
    dropout : float
        Dropout rate.
    pooling : str
        Aggregation strategy across regions.
    """

    def __init__(
        self,
        in_dim: int = 2560,
        hidden_dim: int = 1280,
        proj_dim: int = 256,
        region_repr_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.25,
        pooling: Literal["sum", "mean", "max"] = "sum",
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # ── Shared patch encoder (same architecture as PatchExpert) ───
        self.patch_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

        # ── Per-class gated attention (operates inside each region) ───
        #    Produces (N_patches_in_region, n_classes) scores.
        self.intra_region_attn = GatedAttn(
            input_dim=proj_dim,
            hidden_dim=proj_dim // 2,
            n_classes=num_classes,
        )

        # ── Region representation MLP ────────────────────────────────
        #    Input  = concat( gated_attn_vector (proj_dim),
        #                     mean_patch_embds  (proj_dim) )
        #    Output = region_repr_dim
        self.region_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, region_repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Cross-region additive MIL ─────────────────────────────────
        self.mil = ClassConditionalAdditiveMIL(
            in_dim=region_repr_dim,
            num_classes=num_classes,
            key_dim=region_repr_dim // 2,
            dropout=dropout,
            pooling=pooling,
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _scatter_mean(
        h: torch.Tensor,
        region_ids: torch.Tensor,
        n_regions: int,
    ) -> torch.Tensor:
        """
        Compute per-region mean of patch embeddings using scatter.

        Parameters
        ----------
        h : Tensor ``(N, D)``
            All patch embeddings in the slide.
        region_ids : Tensor ``(N,)`` int
            Which region each patch belongs to (contiguous 0…R-1).
        n_regions : int
            Total number of regions *R*.

        Returns
        -------
        Tensor ``(R, D)``  – per-region mean embedding.
        """
        D = h.shape[1]

        # cast to float32 to avoid problems with mixed precision
        h_fp32 = h.float()

        # Accumulate sums and counts per region
        region_sums = torch.zeros(n_regions, D, device=h.device, dtype=torch.float32)
        region_sums.scatter_add_(0, region_ids.unsqueeze(1).expand_as(h_fp32), h_fp32) # if mixed precision is enabled make sure these are kept to FP32

        counts = torch.zeros(n_regions, device=h.device, dtype=torch.float32)
        counts.scatter_add_(0, region_ids, torch.ones(h.shape[0], device=h.device, dtype=torch.float32)) # if mixed precision is enabled make sure these are kept to FP32
        counts = counts.clamp(min=1)  # avoid /0

        return (region_sums / counts.unsqueeze(1)).to(h.dtype)  # cast back to original dtype

    # ------------------------------------------------------------------
    # def _build_gated_attn_vectors(
    #     self,
    #     h: torch.Tensor,
    #     region_ids: torch.Tensor,
    #     n_regions: int,
    # ) -> torch.Tensor:
    #     """
    #     For every region, compute a gated-attention-weighted sum of its
    #     patch embeddings (using per-class gated attention scores).

    #     Returns shape ``(R, proj_dim)``.
    #     """
    #     # Raw per-class attention scores for all patches in the slide
    #     attn_raw = self.intra_region_attn(h)  # (N, C)

    #     # We need to softmax *within each region* (dim=0 per region).
    #     # To do this efficiently we mask by region.
    #     D = h.shape[1]
    #     gated_vecs = torch.zeros(n_regions, D, device=h.device, dtype=torch.float32)  # (R, D)

    #     h_fp32 = h.float()  # if mixed precision is enabled make sure these are kept to FP32

    #     for r in range(n_regions):
    #         # Boolean mask for patches belonging to region r
    #         rmask = region_ids == r  # (N,)
    #         if rmask.sum() == 0:
    #             continue  # empty region (shouldn't happen normally)

    #         # Softmax attention across patches in this region
    #         region_attn = F.softmax(attn_raw[rmask].float(), dim=0)  # (Nr, C)

    #         # Average attention across classes → single weight per patch
    #         region_attn_avg = region_attn.mean(dim=1)  # (Nr,)

    #         # Attention-weighted sum of patch embeddings for this region
    #         gated_vecs[r] = (region_attn_avg.unsqueeze(1) * h_fp32[rmask]).sum(dim=0)

    #     return gated_vecs.to(h.dtype)  # (R, D)
    def _build_gated_attn_vectors(
        self,
        h: torch.Tensor,          # (N, D)
        region_ids: torch.Tensor, # (N,)  int, values 0…R-1
        n_regions: int,
    ) -> torch.Tensor:
        """Vectorised per-region softmax attention."""
        C = self.num_classes
        attn_raw = self.intra_region_attn(h).float()  # (N, C)
        idx_exp  = region_ids.unsqueeze(1).expand_as(attn_raw)  # (N, C)

        # ── Per-region max for numerical stability ───────────────────
        region_max = torch.full(
            (n_regions, C), float('-inf'),
            device=h.device, dtype=torch.float32
        )
        region_max.scatter_reduce_(0, idx_exp, attn_raw, reduce='amax', include_self=True)
        shifted    = attn_raw - region_max[region_ids]      # (N, C)  subtract region max

        # ── Softmax numerator / denominator ─────────────────────────
        exp_attn   = shifted.exp()                           # (N, C)
        region_sum = torch.zeros(n_regions, C, device=h.device, dtype=torch.float32)
        region_sum.scatter_add_(0, idx_exp, exp_attn)
        softmax_attn = exp_attn / region_sum[region_ids].clamp(min=1e-9)  # (N, C)

        # ── Class-average → per-patch scalar weight ──────────────────
        avg_weight  = softmax_attn.mean(dim=1, keepdim=True)   # (N, 1)
        weighted_h  = (avg_weight * h.float())                  # (N, D)

        # ── Scatter-sum into (R, D) ──────────────────────────────────
        idx_h      = region_ids.unsqueeze(1).expand_as(weighted_h)
        gated_vecs = torch.zeros(n_regions, h.shape[1], device=h.device, dtype=torch.float32)
        gated_vecs.scatter_add_(0, idx_h, weighted_h)

        return gated_vecs.to(h.dtype)   # (R, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_patches: torch.Tensor,
        region_ids: torch.Tensor,
        n_regions: int,
        return_region_emb: bool = False,
    ) -> ExpertOutput:
        """
        Parameters
        ----------
        h_patches : Tensor ``(N, D_raw)``
            Raw patch embeddings (Virchow2 / UNI2) for the whole slide.
        region_ids : Tensor ``(N,)`` int
            Region index (0…R-1) each patch belongs to.
        n_regions : int
            Total number of regions R in this slide.

        Returns
        -------
        ExpertOutput with per-region attention and contributions.
        If ``return_region_emb=True``, also includes region-level embeddings.
        """
        # 1) Project every patch to hidden space ──────────────────────
        h = self.patch_encoder(h_patches)  # (N, proj_dim)

        # 2) Per-region mean patch embedding ──────────────────────────
        mean_embds = self._scatter_mean(h, region_ids, n_regions)  # (R, proj_dim)

        # 3) Per-region gated-attention-weighted sum ──────────────────
        gated_vecs = self._build_gated_attn_vectors(h, region_ids, n_regions)  # (R, proj_dim)

        # 4) Concatenate and project → region representations ─────────
        #    (R, proj_dim*2)  →  (R, region_repr_dim)
        region_repr = self.region_mlp(
            torch.cat([gated_vecs, mean_embds], dim=1)
        )  # (R, region_repr_dim)

        # 5) Cross-region additive MIL ─────────────────────────────────
        logits, attn_w, contribs, bag_repr = self.mil(region_repr)

        if return_region_emb:
            # If requested, we can also return the region-level embeddings
            return RegionExpertOutput(
                    logits=logits,
                    attn_weights=attn_w,
                    contributions=contribs,
                    representation=bag_repr,
                    region_embeddings=region_repr,
            )

        return ExpertOutput(
            logits=logits,
            attn_weights=attn_w,
            contributions=contribs,
            representation=bag_repr,
        )


# ======================================================================
# 4.  CellExpert
# ======================================================================

class CellExpert(nn.Module):
    """
    MIL expert operating at the **cell / sub-patch** level.

    Workflow
    --------
    1. Receive the **top-k** 256×256 patch indices selected by
       ``PatchExpert`` (based on highest attention).
    2. All cell tokens from the K patches are concatenated and run
       through ``ClassConditionalAdditiveMIL`` to produce cell-level
       logits, attention, contributions and a bag representation.

    Parameters
    ----------
    token_dim : int
        Dimensionality of each dense sub-patch token produced by the
        frozen backbone (e.g. 1280 for UNI2, 2560 for Virchow2).
    proj_dim : int
        Projected dimension of cell tokens before additive MIL.
    num_classes : int
        Number of target classes.
    dropout : float
        Dropout rate in the projection MLP.
    pooling : str
        Aggregation strategy for cell-level MIL.
    """

    def __init__(
        self,
        token_dim: int = 1280,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.25,
        pooling: Literal["sum", "mean", "max"] = "sum",
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.token_dim = token_dim

        # ── Cell-token encoder: token_dim → proj_dim ─────────────────
        #    Projects the dense ViT tokens into a lower-dimensional
        #    space suitable for the attention + classifier heads.
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

        # ── Additive MIL across all cell tokens ─────────────────────
        self.mil = ClassConditionalAdditiveMIL(
            in_dim=proj_dim,
            num_classes=num_classes,
            key_dim=proj_dim // 2,
            dropout=dropout,
            pooling=pooling,
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_cell_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> ExpertOutput:
        """
        Parameters
        ----------
        h_cell_tokens : Tensor ``(K * T, token_dim)`` or ``(K, T, token_dim)``
            Pre-extracted dense tokens.
        mask : Tensor ``(K * T,)`` bool, optional
            True = valid token.

        Returns
        -------
        ExpertOutput
        """
        if h_cell_tokens.dim() == 3:
            K, T, Dt = h_cell_tokens.shape
            h_cell_tokens = h_cell_tokens.reshape(K * T, Dt)

        h = self.encoder(h_cell_tokens)  # (K*T, proj_dim)

        # 2) Additive MIL over all cell tokens ────────────────────────
        logits, attn_w, contribs, bag_repr = self.mil(h, mask=mask)

        return ExpertOutput(
            logits=logits,
            attn_weights=attn_w,
            contributions=contribs,
            representation=bag_repr,
        )


# ======================================================================
# 5.  HierarchicalMIL  – Fusion of the three experts
# ======================================================================

class HierarchicalMIL(nn.Module):
    """
    Hierarchical Multi-Expert MIL for whole-slide classification.

    Fuses ``PatchExpert``, ``RegionExpert`` and ``CellExpert`` outputs
    via **learnable expert alphas** computed from each expert's
    sqrt-N normalised logits.

    The final slide-level logits are computed as::

        region_norm = region_logits / sqrt(N_regions)
        patch_norm  = patch_logits  / sqrt(N_patches)
        cell_norm   = cell_logits   / sqrt(N_cells)

        alphas      = softmax([region_norm, patch_norm, cell_norm], dim=-1)
        final_logits = alpha_r * region_norm
                     + alpha_p * patch_norm
                     + alpha_c * cell_norm          # (n_classes,)

    Parameters
    ----------
    input_dim : int
        Raw patch embedding dim (Virchow2=2560, UNI2=1024).
    hidden_dim : int
        Intermediate projection size in the expert encoders.
    hidden_dim_2 : int
        Instance representation dim (proj_dim) inside each expert.
    n_classes : int
        Number of target classes.
    dropout : float
        Dropout rate.
    pooling : str
        Additive-MIL pooling strategy across instances.
    top_k : int
        Number of high-attention patches sent to CellExpert.
    cell_token_dim : int
        Dim of each dense ViT token (UNI2=1536, Virchow2=1280).
    instance_batch_size : int
        Kept for API compat with ``MILTrainer`` (unused here –
        batching happens inside each expert as needed).
    """

    def __init__(
        self,
        region_in_dim: int = 2560,
        patch_in_dim: int = 1280,
        cell_token_dim: int = 1280,
        hidden_dim_2: int = 256,
        n_classes: int = 1,
        dropout: float = 0.25,
        pooling: Literal["sum", "mean", "max"] = "sum",
        top_k: int = 512,
        instance_batch_size: int = 1024,  # API compat
        scale_drop_p: float = 0.15,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.top_k = top_k
        self.scale_dropout = scale_drop_p

        # ── 1. Patch Expert ──────────────────────────────────────────
        self.patch_expert = PatchExpert(
            in_dim=patch_in_dim,
            hidden_dim=patch_in_dim//2,
            proj_dim=hidden_dim_2,
            num_classes=n_classes,
            dropout=dropout,
            pooling=pooling,
            top_k=top_k,
        )

        # ── 2. Region Expert ─────────────────────────────────────────
        self.region_expert = RegionExpert(
            in_dim=region_in_dim,
            hidden_dim=region_in_dim//2,
            proj_dim=hidden_dim_2,
            region_repr_dim=hidden_dim_2,
            num_classes=n_classes,
            dropout=dropout,
            pooling=pooling,
        )

        # ── 3. Cell Expert ───────────────────────────────────────────
        self.cell_expert = CellExpert(
            token_dim=cell_token_dim,
            hidden_dim=cell_token_dim//2,
            proj_dim=hidden_dim_2,
            num_classes=n_classes,
            dropout=dropout,
            pooling=pooling,
        )

        # ── 4. Expert gating ─────────────────────────────────────
        #    Alphas are computed in forward() from the sqrt-N normalised
        #    logits of each expert — no learnable gating parameters needed.

    # ------------------------------------------------------------------
    def forward(
        self,
        h_patches: torch.Tensor,
        region_ids: torch.Tensor,
        n_regions: int,
        h_region_features: Optional[torch.Tensor] = None,
        h_cell_tokens: Optional[torch.Tensor] = None,
        attn_return: bool = False,
        return_region_emb: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[Dict], Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        h_patches : Tensor ``(1, N, 1280)``  or  ``(N, 1280)``
            CLS token per patch — PatchExpert input.
            A leading batch dim of 1 is squeezed automatically.
        region_ids : Tensor ``(N,)`` int
            Region assignment for every patch (0 … R-1).
        n_regions : int
            Number of regions R.
        h_region_features : Tensor ``(1, N, 2560)``  or  ``(N, 2560)``, optional
            CLS+mean(patch tokens) per patch — RegionExpert input.
            If None, falls back to h_patches.
        h_cell_tokens : Tensor ``(1, N, 256, D)``  or  ``(N, 256, D)``, optional
            Dense token grids per patch — CellExpert input.
            Top-K patches are indexed after PatchExpert selects them.
        attn_return : bool
            If True, return per-expert attention dicts alongside logits.

        Returns
        -------
        logits : Tensor ``(n_classes,)``
            Final fused slide-level logits.
        attn_dict : dict or None
            ``{expert_name: ExpertOutput}`` when ``attn_return=True``.
        contributions_dict : dict or None
            Same, but keyed to contributions tensors.
        region_embeddings : Tensor or None
            Region-level embeddings when ``return_region_emb=True``.
        """
        # ── Squeeze optional batch dim ───────────────────────────────
        if h_patches.dim() == 3:
            h_patches = h_patches.squeeze(0)  # (N, 1280)

        # ── A.  Patch Expert ─────────────────────────────────────────
        patch_out, top_k_idx = self.patch_expert(h_patches)

        # ── B.  Region Expert ────────────────────────────────────────
        # Use h_region_features (CLS+mean, 2560-dim) when available;
        # fall back to h_patches if not supplied.
        _h_region = h_region_features if h_region_features is not None else h_patches
        if _h_region.dim() == 3:
            _h_region = _h_region.squeeze(0)
        region_out = self.region_expert(_h_region, region_ids, n_regions, return_region_emb=return_region_emb)

        # ── C.  Cell Expert ──────────────────────────────────────────
        #    Index pre-extracted cell tokens from H5.
        if h_cell_tokens is not None:
            if h_cell_tokens.dim() == 4:
                h_cell_tokens = h_cell_tokens.squeeze(0)  # (N, 256, D)
            device = h_patches.device
            cell_tokens_topk = h_cell_tokens[top_k_idx]  # (K, 256, D)
            cell_out = self.cell_expert(h_cell_tokens=cell_tokens_topk.to(device))
        else:
            device = h_patches.device
            cell_out = ExpertOutput(
                logits=torch.zeros(self.n_classes, device=device),
                attn_weights=torch.zeros(1, self.n_classes, device=device),
                contributions=torch.zeros(1, self.n_classes, device=device),
                representation=torch.zeros(
                    self.patch_expert.proj_dim, device=device
                ),
            )

        # ── C'.  Scale Dropout (training only) ───────────────────────
        #    Independently drop each expert with probability scale_drop_p.
        #    At least one expert always survives.  Dropped experts get
        #    zeroed representations in the fusion input and −∞ gating
        #    logits (→ 0 weight after softmax), so they contribute nothing
        #    to the final prediction.  ExpertOutput objects are kept
        #    intact so deep-supervision aux loss can still back-prop
        #    through every expert's individual logits.
        device = h_patches.device
        if self.training and self.scale_dropout > 0.0:
            keep_mask = torch.bernoulli(
                torch.full((3,), 1.0 - self.scale_dropout, device=device)
            )  # (3,)  1=keep, 0=drop
            # Guarantee at least one survivor
            if keep_mask.sum() == 0:
                survivor = torch.randint(0, 3, (1,), device=device)
                keep_mask[survivor] = 1.0
        else:
            keep_mask = torch.ones(3, device=device)
        
        # Synchronize mask aacross DPP ranks
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(keep_mask, src=0)

        # ── D.  Fusion ───────────────────────────────────────────────
        # D1) Instance counts from attn_weights — path-agnostic (works for
        #     and cell warmup curriculum).
        region_n = region_out.attn_weights.shape[0]
        patch_n  = patch_out.attn_weights.shape[0]
        cell_n   = cell_out.attn_weights.shape[0]

        # D2) Normalise each expert's logits by sqrt(N_instances) to remove
        #     the sum-pooling scale dependency on bag size.
        region_logit_norm = region_out.logits / (region_n ** 0.5 + 1e-8)
        patch_logit_norm  = patch_out.logits  / (patch_n  ** 0.5 + 1e-8)
        cell_logit_norm   = cell_out.logits   / (cell_n   ** 0.5 + 1e-8)

        # D3) Logit-based expert gating: each expert votes on its own weight
        #     via its normalised logit.  Dropped experts receive -inf before
        #     softmax so they contribute 0 weight after renormalisation.
        expert_logits_stack = torch.stack(
            [region_logit_norm, patch_logit_norm, cell_logit_norm], dim=-1
        )  # (n_classes, 3)
        drop_penalty = (1.0 - keep_mask).unsqueeze(0) * (-1e9)  # (1, 3)
        alphas = F.softmax(
            expert_logits_stack + drop_penalty, dim=-1
        )  # (n_classes, 3)
        alphas = alphas.transpose(0, 1)  # (3, n_classes)

        # D4) Weighted sum of normalised expert logits
        logits = (
            alphas[0] * region_logit_norm
            + alphas[1] * patch_logit_norm
            + alphas[2] * cell_logit_norm
        )  # (n_classes,)

        # ── E.  Return ───────────────────────────────────────────────
        if attn_return:
            attn_dict = {
                "region": region_out,
                "patch": patch_out,
                "cell": cell_out,
                'gating_alphas': alphas,
                'scale_keep_mask': keep_mask,  # (3,) 1=kept, 0=dropped
                'n_instances': {'region': region_n, 'patch': patch_n, 'cell': cell_n},
            }
            return logits, attn_dict

        return logits, None
