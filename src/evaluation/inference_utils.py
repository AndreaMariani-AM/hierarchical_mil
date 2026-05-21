"""
Inference utilities for MIL models.

Provides notebook-friendly helpers to:
    1. Load a trained ``MILTrainer`` checkpoint.
    2. Run inference over a fold split via ``DataLoader`` +
       ``lightning.Trainer.predict()`` and return logits + a normalised
       attention dict for every slide.
    3. Unpack the raw attention output from different model types into a
       *consistent nested structure* so downstream visualization code
       can use the same key regardless of model type.
    4. Re-derive top-k patch indices from PatchExpert attention.

Supported models
----------------
* ``HierarchicalMIL``   → hierarchical H5 file (``mid/H_patch``,
  ``mid/H_region``, ``cells/H``, ``mid/region_id``, ``regions/…``)
* ``AdditiveMIL`` / ``AttentionMIL`` → hierarchical H5 file
  (reads ``mid/H_region``; spatial coordinates preserved).

Usage example
-------------
::

    from src.evaluation.inference_utils import load_model, predict_slide

    model   = load_model("experiments/training/hierarchical/fold_0-….ckpt")
    results = predict_slide(model, "data/folds/fold_0/fold_0.csv",
                            "data/features/.../virchow2_128_hierarchical",
                            split="val")

    # results — list[dict], one entry per slide
    # dict keys: logits, prob, label, slide_id, attn_dict
    # attn_dict keys (hierarchical): patch, region, cell, gating_alphas,
    #   scale_keep_mask, n_instances
    # attn_dict keys (additive/attention): patch
    #   patch sub-keys: attn_weights (N, C), contributions (N, C)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

# ── project imports ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.trainer import MILTrainer


# ======================================================================
# 1.  load_model
# ======================================================================

def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> MILTrainer:
    """Load a trained ``MILTrainer`` from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the ``.ckpt`` file produced by Lightning.
    device : str, optional
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-selects CUDA when available).

    Returns
    -------
    MILTrainer
        Model in ``eval`` mode on the chosen device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    map_device = torch.device(device)

    model = MILTrainer.load_from_checkpoint(
        str(checkpoint_path),
        map_location=map_device,
        strict=False,
    )
    model.eval()
    model.to(map_device)
    return model


# ======================================================================
# 2.  predict_slide  (dispatches by model type + file format)
# ======================================================================

def predict_slide(
    model: MILTrainer,
    fold_csv: Union[str, Path],
    h5_dir: Union[str, Path],
    split: str = "val",
    activate_cell_expert: bool = True,
    accelerator: str = "auto",
    devices: int = 1,
) -> List[Dict]:
    """Run inference over a fold split and return predictions for all slides.

    Uses ``DataLoader`` + ``lightning.Trainer.predict()`` for device-safe
    inference, fully consistent with the training pipeline.

    Parameters
    ----------
    model : MILTrainer
        Loaded model from :func:`load_model`.
    fold_csv : str or Path
        Path to a fold CSV with columns ``Slide``, ``Condition``,
        ``Feature_Path``, ``split``, ``use_slide``.
    h5_dir : str or Path
        Directory containing per-slide hierarchical H5 files named
        ``{slide_id}.h5``.
    split : str
        Which split to run on — ``"val"``, ``"train"``, or another label
        present in the CSV's ``split`` column.
    activate_cell_expert : bool
        Temporarily set ``cell_warmup_start = 0`` so the CellExpert
        fires during prediction, regardless of the training epoch at
        which the checkpoint was saved.  Only relevant for hierarchical
        models.
    accelerator : str
        Lightning accelerator string — ``"auto"``, ``"gpu"``, ``"cpu"``.
    devices : int
        Number of devices to use.

    Returns
    -------
    list[dict]
        One dict per slide (in dataset order) with keys:

        ``logits``    — :class:`torch.Tensor` ``(n_classes,)`` on CPU.
        ``prob``      — ``float`` (sigmoid) or ``list[float]`` (softmax).
        ``label``     — :class:`torch.Tensor` scalar on CPU.
        ``slide_id``  — ``str``.
        ``attn_dict`` — normalised dict; see :func:`unpack_attn_dict`.
    """
    import lightning as L
    from torch.utils.data import DataLoader
    from src.data.dataset import (
        HierarchicalRepresentationsDataset,
        hierarchical_collate_fn,
        RepresentationsDataset,
    )

    fold_csv = Path(fold_csv)
    h5_dir   = Path(h5_dir)

    if model._is_hierarchical:
        dataset    = HierarchicalRepresentationsDataset(
            csv_path=str(fold_csv),
            h5_dir=str(h5_dir),
            split=split,
        )
        collate_fn = hierarchical_collate_fn
    else:
        dataset    = RepresentationsDataset(
            csv_path=str(fold_csv),
            representation_dir=str(h5_dir),
            split=split,
        )
        collate_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Temporarily override cell_warmup_start so that the cell_expert_active
    # property returns True during predict_step (current_epoch == 0 outside
    # of a training loop).
    _orig_warmup = model.cell_warmup_start
    if activate_cell_expert and model._is_hierarchical:
        model.cell_warmup_start = 0

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    try:
        raw_results = trainer.predict(model, dataloaders=dataloader)
    finally:
        model.cell_warmup_start = _orig_warmup

    # Normalise attn_dict structure and add probability
    model_type = "hierarchical" if model._is_hierarchical else "additive"
    results = []
    for r in raw_results:
        attn_dict = unpack_attn_dict(r["attn_dict"], model_type=model_type)
        logits    = r["logits"]
        prob = (
            torch.sigmoid(logits).item()
            if model.n_classes == 1
            else torch.softmax(logits, dim=-1).tolist()
        )
        results.append({
            "logits":    logits,
            "prob":      prob,
            "label":     r["label"],
            "slide_id":  r["slide_id"],
            "attn_dict": attn_dict,
        })

    return results


# ======================================================================
# 3.  unpack_attn_dict  – consistent structure across model types
# ======================================================================

def unpack_attn_dict(
    raw_attn_dict,
    model_type: str = "hierarchical",
) -> Dict:
    """Normalise the raw attention output into a consistent nested dict.

    All models expose a ``'patch'`` key with ``attn_weights`` and
    ``contributions`` tensors of shape ``(N, n_classes)`` on CPU.
    Hierarchical models additionally expose ``'region'``, ``'cell'``,
    ``'gating_alphas'``, ``'scale_keep_mask'``, and ``'n_instances'``.

    Conversion rules
    ----------------
    * ``ExpertOutput`` dataclasses are exploded to plain ``dict``.
    * All :class:`torch.Tensor` values are detached and moved to CPU.
    * ``"additive"`` / ``"attention"`` outputs (flat tensors under
      keys ``'attn_weights'`` / ``'contributions'``) are wrapped under
      a ``'patch'`` sub-dict.

    Parameters
    ----------
    raw_attn_dict : dict
        Direct output of ``model.forward(attn_return=True)`` or the
        already-detached dict from ``MILTrainer._detach_attn_val``.
    model_type : str
        ``"hierarchical"`` or ``"additive"`` / ``"attention"``.

    Returns
    -------
    dict
        Normalised structure guaranteed to have ``result['patch']``
        with ``attn_weights`` and ``contributions``.
    """
    def _to_cpu(v):
        """Recursively detach tensors and expand dataclass fields."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if hasattr(v, "__dataclass_fields__"):
            return {
                field: (
                    getattr(v, field).detach().cpu()
                    if isinstance(getattr(v, field), torch.Tensor)
                    else getattr(v, field)
                )
                for field in v.__dataclass_fields__
            }
        if isinstance(v, dict):
            return {k2: _to_cpu(v2) for k2, v2 in v.items()}
        return v

    if model_type == "hierarchical":
        return {k: _to_cpu(v) for k, v in raw_attn_dict.items()}

    # additive / attention — wrap flat tensors under 'patch'
    return {
        "patch": {
            "attn_weights": _to_cpu(raw_attn_dict.get("attn_weights")),
            "contributions": _to_cpu(raw_attn_dict.get("contributions")),
        }
    }


# ======================================================================
# 4.  get_top_k_patch_indices
# ======================================================================

def get_top_k_patch_indices(attn_dict: Dict, k: int) -> torch.Tensor:
    """Re-derive the top-k patch indices from PatchExpert attention weights.

    Reproduces the same selection logic as ``PatchExpert.forward`` without
    requiring changes to ``experts_MIL.py``.

    Parameters
    ----------
    attn_dict : dict
        Normalised attention dict (output of :func:`unpack_attn_dict`).
    k : int
        Number of top-attended patches to return.

    Returns
    -------
    Tensor ``(min(k, N),)`` int64
        Indices of the top-k patches by mean attention across classes,
        sorted ascending (safe for positional indexing into arrays).
    """
    patch_attn = attn_dict["patch"]["attn_weights"]  # (N, n_classes)
    mean_attn = patch_attn.mean(dim=-1)              # (N,)
    effective_k = min(k, mean_attn.shape[0])
    top_k_idx = torch.topk(mean_attn, effective_k).indices
    return top_k_idx.sort().values                   # ascending order
