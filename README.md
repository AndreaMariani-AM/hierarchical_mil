# Hierarchical Expert MIL Classification from Whole-Slide Images

> **Goal**: Classify Whole-Slide Images (WSI) using a hierarchical
> multi-expert Multiple Instance Learning (MIL) framework that reasons
> across three spatial scales: **regions** (2048×2048), **patches**
> (256×256) and **cells** (sub-patch 16×16 dense token grids).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
   - [Spatial Hierarchy](#spatial-hierarchy)
   - [Core Building Block: ClassConditionalAdditiveMIL](#core-building-block-classconditionaladditivemil)
   - [Three Experts](#three-experts)
   - [Fusion](#fusion)
   - [Scale Dropout](#scale-dropout)
4. [Data Pipeline](#data-pipeline)
   - [Step 1 – Tissue Segmentation](#step-1--tissue-segmentation)
   - [Step 2 – Hierarchical Tiling](#step-2--hierarchical-tiling)
   - [Step 3 – Feature Extraction](#step-3--feature-extraction)
   - [Step 4 – H5 Storage Format](#step-4--h5-storage-format)
5. [Training](#training)
   - [Fold Creation](#fold-creation)
   - [MIL Training](#mil-training)
   - [Loss Function and Deep Supervision](#loss-function-and-deep-supervision)
   - [Model Types](#model-types)
6. [Configuration](#configuration)
7. [Setup & Installation](#setup--installation)
8. [Running the Pipeline](#running-the-pipeline)
9. [Key Design Decisions](#key-design-decisions)
10. [Nuclei Identification (legacy)](#nuclei-identification-legacy)

---

## Overview

Standard MIL treats a WSI as a flat bag of patch embeddings. This
project extends that paradigm by introducing **three specialised
experts** that each attend to a different spatial resolution:

| Expert | Scale | Input | Role |
|--------|-------|-------|------|
| **RegionExpert** | 2048×2048 | Aggregated patch embeddings per region | Captures tissue-architecture patterns |
| **PatchExpert** | 256×256 | Foundation-model embeddings | Captures cell-neighbourhood / crypt patterns |
| **CellExpert** | Sub-patch tokens | Pre-extracted dense ViT tokens from H5 (`/cells/H`) | Captures single-cell morphology |

Their outputs are fused via **learnable expert alphas** computed from each
expert's sqrt-N normalised logits via softmax gating to produce a single
slide-level prediction.

---

## Project Structure

```
Predictive_model/
├── configs/                        # YAML configuration files
│   ├── base_config.yaml            #   training & fold config
│   └── feature_extraction.yaml     #   segmentation / tiling / FM config
│
├── data/
│   ├── annotations/                # Segmentation masks, nuclei annotations
│   ├── features/extracted_features/# Per-slide H5 files with embeddings
│   ├── folds/                      # 5-fold train/val CSVs
│   ├── metadata/                   # Cohort CSVs (slide paths + labels)
│   └── processed/                  # Slide-level metadata
│
├── scripts/                        # Executable pipeline scripts
│   ├── feature_extraction.py       #   Seg → Tile → Extract → Save H5
│   ├── create_folds.py             #   Stratified k-fold splits
│   ├── train_MIL.py                #   Lightning training entry-point
│   ├── evaluate.py                 #   Evaluation & metrics
│   └── inference.py                #   Run trained model on new slides
│
├── sbatchers/                      # SLURM job scripts
│   ├── run_feature_extraction.sh   #   GPU array job for extraction
│   └── run_MIL_training.sh         #   GPU array job for 5-fold training
│
├── src/                            # Reusable library modules
│   ├── data/
│   │   ├── dataset.py              #   RepresentationsDataset / HierarchicalRepresentationsDataset
│   │   ├── preprocessing.py        #   Tiling, filtering, H5 save/load
│   │   └── augmentation.py         #   Data augmentations
│   │
│   ├── models/
│   │   ├── attention.py            #   GatedAttn, AttnVanilla, MHA
│   │   ├── modules.py              #   PatchEmbed, TransformerBlock, MLP
│   │   ├── MIL.py                  #   AttentionMIL, AdditiveMIL (+ re-exports HierarchicalMIL)
│   │   ├── experts_MIL.py          #   ★ PatchExpert, RegionExpert, CellExpert, HierarchicalMIL
│   │   └── ViT.py                  #   UNI2Dense, Virchow2Dense (frozen backbones)
│   │
│   └── training/
│       └── trainer.py              #   MILTrainer (Lightning module)
│
├── notebooks/                      # Jupyter notebooks (exploration / analysis)
├── experiments/                    # Checkpoints, TensorBoard logs
├── results/                        # Figures, reports, tables
└── requirements.txt
```

---

## Architecture

### Spatial Hierarchy

A single WSI is decomposed into a three-level hierarchy:

```
                    ┌─────────────────────────────────┐
                    │         Whole Slide Image        │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │    Regions (2048×2048 @ 0.5 mpp) │  ← RegionExpert
                    │    Tiled over tissue, filtered   │
                    └────────────────┬────────────────┘
                                     │  each region contains ~64 patches
                    ┌────────────────▼────────────────┐
                    │    Patches (256×256 @ 0.5 mpp)   │  ← PatchExpert
                    │    Sub-tiles within each region  │
                    └────────────────┬────────────────┘
                                     │  top-K highest-attention patches
                    ┌────────────────▼────────────────┐
                    │    Cell tokens (16×16 grid)       │  ← CellExpert
                    │    Pre-extracted dense ViT tokens │
                    │    (256 per patch) loaded from H5 │
                    │    /cells/H (N, 256, token_dim)   │
                    └─────────────────────────────────┘
```

---

### Core Building Block: ClassConditionalAdditiveMIL

All three experts share this reusable module (`src/models/experts_MIL.py`).
Given a set of instance embeddings `h` of shape `(N, D)` it computes:

1. **Per-class gated attention** — `GatedAttn(tanh ⊙ sigmoid → n_classes)`
   produces raw scores `(N, C)`, softmaxed across the N instances per class.
2. **Optional masking** — padded / invalid instances receive `−∞` attention
   before softmax, contributing zero to the bag.
3. **Per-class instance classifiers** — one `Linear(D → 1)` head per class.
4. **Contributions** — `attn_weight[i,c] × instance_pred[i,c]` for every
   instance × class, shape `(N, C)`. Directly interpretable as per-instance
   impact on the final prediction.
5. **Pooling** — contributions are pooled across instances (sum / mean / max,
   configurable) → bag logits `(C,)`.
6. **Bag representation** — attention-weighted mean of instance embeddings
   `(D,)`, used downstream in the fusion head.

The `ExpertOutput` named-tuple standardises what every expert returns:
`(logits, attn_weights, contributions, representation)`.

---

### Three Experts

#### PatchExpert

Processes all 256×256 patch embeddings for the whole slide. Also selects
the top-K most-attended patches (by class-averaged attention) and exposes
their indices for the CellExpert to zoom into.

```
Raw patch embeddings (N, 2560)
        │
  ┌─────▼──────┐
  │  Encoder   │   Linear(2560→1280) → ReLU → Dropout(0.25) → Linear(1280→256)
  └─────┬──────┘
        │  (N, 256)
  ┌─────▼──────────────────┐
  │ ClassConditionalAddMIL │   → logits (C,) + attn_weights (N, C)
  └─────┬──────────────────┘     + contributions (N, C) + bag_repr (256,)
        │
        ├──► ExpertOutput
        │
        └──► top_k_indices (K,)   ← avg attention across classes → torch.topk
```

Default K = 128–512 (configurable via `--top_k`).

---

#### RegionExpert

Builds one representation per 2048×2048 region by combining a simple mean
of its patch embeddings with a gated-attention-weighted sum, then runs
additive MIL across all regions.

```
All patch embeddings (N, 2560)
        │
  ┌─────▼──────┐
  │  Encoder   │   Same MLP architecture as PatchExpert
  └─────┬──────┘
        │  (N, 256)  — split by region_id
        │
   For each region r  (N_r patches):
   ┌────┴──────────────────────────────────────┐
   │                                            │
   ▼                                            ▼
mean(patches_r) → (256,)        GatedAttn, softmax within region r
                                 → attn-weighted sum → (256,)
   │                                            │
   └──────────────► concat ◄───────────────────┘
                       │  (512,)
                 ┌─────▼──────┐
                 │ Region MLP │   Linear(512→256) → ReLU → Dropout
                 └─────┬──────┘
                       │  region_repr_r (256,)

Region representations (R, 256)
        │
  ┌─────▼──────────────────┐
  │ ClassConditionalAddMIL │   → ExpertOutput
  └────────────────────────┘
```

The per-region representation captures both **average tissue composition**
(mean pooling) and **locally salient features** (gated attention) within
each spatial block.

---

#### CellExpert

Receives the top-K patch indices from PatchExpert, indexes pre-extracted
dense ViT tokens from H5 (`/cells/H`), and runs additive MIL over all
cell tokens.  Token extraction is performed **once offline** by
`scripts/feature_extraction.py` using a frozen foundation model; at
training time CellExpert only reads and processes the stored tensors.

```
top_k_indices (K,)  ←  from PatchExpert
        │
   ┌────▼────────────────────────────────┐
   │  Index pre-extracted cell tokens    │
   │  h_cell_tokens[top_k_idx]           │
   │  → (K, 256, token_dim)              │
   │  i.e. 16×16 spatial grid per patch  │
   └────┬────────────────────────────────┘
        │  Flatten → (K×256, token_dim)
  ┌─────▼──────┐
  │  Encoder   │   Linear(token_dim → 256) → ReLU → Dropout
  └─────┬──────┘
        │  (K×256, 256)
  ┌─────▼──────────────────┐
  │ ClassConditionalAddMIL │   → ExpertOutput
  └────────────────────────┘
```

**Note:** Cell tokens are pre-extracted by `scripts/feature_extraction.py`
using a frozen `Virchow2Dense` or `UNI2Dense` backbone (lazyslide wrappers).
Token dimensions: **Virchow2** → 1280-d, **UNI2** → 1536-d.  No backbone
inference occurs at training time.

---

### Fusion

After all three experts produce their outputs, `HierarchicalMIL` fuses
them via **logit-based expert gating** — no separate fusion MLP:

```
region_logits (C,) / sqrt(N_regions)  ──┐
patch_logits  (C,) / sqrt(N_patches)  ──┼──► stack → (n_classes, 3)
cell_logits   (C,) / sqrt(N_cells)    ──┘
                                               │
                               + drop_penalty (−∞ for dropped experts)
                                               │
                    ┌──────────▼──────────┐
                    │  softmax(dim=-1)     │   alphas (n_classes, 3)
                    └──────────┬──────────┘   ← per-class expert weights
                               │  transpose → (3, n_classes)
                               │
         ┌─────────────────────▼──────────────────────┐
         │  final_logits = α₀·region_norm             │
         │               + α₁·patch_norm              │
         │               + α₂·cell_norm               │
         │               = (n_classes,)               │
         └─────────────────────┬──────────────────────┘
                               │
                    BCEWithLogitsLoss / CrossEntropyLoss
```

**Expert gating:** Each expert votes on its own weight through its
sqrt-N normalised logit.  Scale-dropped experts receive a `−∞` penalty
before the softmax, collapsing their alpha to zero so they contribute
nothing to the final prediction.

---

### Scale Dropout

To prevent co-adaptation among experts and force each one to learn
meaningful independent representations, a **Scale Dropout** mechanism
is applied during training only:

- Each expert is independently dropped with probability `p` (default 0.15,
  configurable via `--scale_drop_p`).
- **At least one expert always survives** — if all are dropped, one is
  randomly re-activated.
- Dropped experts have their **representation zeroed** in the fusion input.
  Surviving experts are rescaled by `3 / n_active` to maintain expected
  magnitude (inverted-dropout style).
- Dropped experts receive `−∞` gating logits → weight collapses to 0 after
  softmax, contributing nothing to weighted logits.
- Crucially, **individual expert logits are preserved intact** so the
  auxiliary deep-supervision loss can still back-propagate through every
  expert regardless of whether it was dropped in the fusion.

Expected drop frequencies at `p = 0.15`:

| Scenario | Probability |
|---|---|
| All 3 experts active | ~61% of steps |
| Exactly 1 expert dropped | ~33% of steps |
| Exactly 2 experts dropped | ~6% of steps |

---

## Data Pipeline

The full pipeline is orchestrated by `scripts/feature_extraction.py`
and configured via `configs/feature_extraction.yaml`.

### Step 1 – Tissue Segmentation

```python
ls.seg.tissue(wsi, key_added="dl_fragments", model="pathprofiler", mpp=5.0)
```

A deep-learning segmentation model (`pathprofiler`) identifies tissue
fragments at 10× the target MPP (to fit large slides in GPU memory).
Segmentation masks are saved as WKT geometries in H5 files under
`data/annotations/seg_masks_hierarchical/`.

### Step 2 – Hierarchical Tiling

Two-level tiling creates the spatial hierarchy:

1. **Region tiles (2048×2048 @ 0.5 mpp)**: generated over the entire
   slide bounding box, then filtered to keep only tiles overlapping
   tissue polygons.

2. **Sub-tiles (256×256 @ 0.5 mpp)**: generated inside each region
   tile. Each sub-tile records which region it belongs to via a
   `region_tiles_id` column.

3. **Tissue filtering**: sub-tiles not overlapping tissue polygons are
   removed.

4. **Re-indexing**: after filtering, region and sub-tile IDs are
   re-indexed to contiguous `0…N-1` so that `region_tiles_id` is a
   valid positional index into the H5 region array.

```python
pp.tile_whole_wsi_with_background_filter(wsi, tile_px=2048, mpp=0.5,
                                          key_added="region_tiles")
pp.filter_tiles_by_tissue_mask(wsi, tile_key="region_tiles")
pp.create_subtiles_from_region_tiles(wsi, subtile_px=256, mpp=0.5,
                                      key_added="tiles")
pp.filter_tiles_by_tissue_mask(wsi, tile_key="tiles")
pp.reindex_tiles_and_regions(wsi)
```

### Step 3 – Feature Extraction

Foundation model embeddings are extracted for every 256×256 patch using
`lazyslide`:

```python
ls.tl.feature_extraction(wsi, model="virchow2", device="cuda", batch_size=512)
```

This produces a 2560-dimensional embedding per patch (Virchow2 default).
These embeddings are stored offline in H5 files and loaded at training
time — the expensive extraction is done **once** before any model training.

### Step 4 – H5 Storage Format

Each slide produces one H5 file with the following layout:

```
{slide_name}.h5
├── /regions/
│   ├── region_id      (R,)            int32    – contiguous 0…R-1
│   └── xy_2048        (R, 4)          int32    – [minx, miny, maxx, maxy]
│
├── /mid/
│   ├── tile_id        (N,)            int32    – contiguous 0…N-1
│   ├── region_id      (N,)            int32    – maps each patch → its region
│   ├── H_patch        (N, 1280)       float16  – Virchow2 CLS token (PatchExpert)
│   └── H_region       (N, 2560)       float16  – Virchow2 CLS+mean (RegionExpert)
│
└── /cells/
    └── H              (N, 256, 1280)  float16  – dense 16×16 tokens (CellExpert)
```

The H5 structure is created **before** feature extraction (metadata only),
then embeddings are written in-place after extraction:

```python
pp.save_tile_region_h5(wsi, outdir, model_name, tile_size, slide_name)
# ... run feature extraction ...
pp.save_embeddings_to_h5(wsi, model_name)
```

---

## Training

### Fold Creation

Stratified 5-fold train/val splits are created by
`scripts/create_folds.py` using `configs/base_config.yaml`:

```bash
python scripts/create_folds.py
```

Each fold CSV (`data/folds/fold_{k}/fold_{k}.csv`) contains columns
`Slide`, `Condition`, `Feature_Path`, and `split` (train / val).

### MIL Training

Training is handled by PyTorch Lightning via `scripts/train_MIL.py`:

```bash
python scripts/train_MIL.py \
    --config_file configs/base_config.yaml \
    --fold 0 \
    --model_type hierarchical \
    --pooling sum \
    --frozen_backbone Virchow2 \
    --top_k 128 \
    --scale_drop_p 0.15 \
    --max_epochs 100 \
    --accumulate_grad_batches 32 \
    --lr 1e-4 \
    --weight_decay 1e-3
```

The `MILTrainer` Lightning module (`src/training/trainer.py`):

- Instantiates the chosen model (`AttentionMIL`, `AdditiveMIL`, or `HierarchicalMIL`).
- Handles loss selection: `BCEWithLogitsLoss` for binary (`n_classes=1`)
  or `CrossEntropyLoss` for multi-class.
- Tracks metrics per epoch: Accuracy, F1, AUROC.
- Supports gradient accumulation (`accumulate_grad_batches=32`),
  gradient clipping (`max_norm=1.0`), early stopping, and best-model
  checkpointing.
- Runs distributed training via DDP across 4 nodes, 1 GPU each.

For hierarchical training, the dataloader returns a 7-tuple:

```python
(h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)
```

The trainer detects `_is_hierarchical=True` and forwards the extra
tensors to `HierarchicalMIL.forward()`.

---

### Loss Function and Deep Supervision

The total training loss combines the primary loss on the fused prediction
with unweighted per-expert auxiliary losses:

```
Loss = Loss_main(final_logits, label)
     + Loss_region(region_logits / sqrt(N_r), label)
     + Loss_patch(patch_logits  / sqrt(N_p), label)
     + Loss_cell(cell_logits    / sqrt(N_c), label)   ← only when CellExpert is active
```

All per-expert logits are sqrt-N normalised before the loss to remove
the scale dependency on bag size, matching the normalisation applied in
the fusion gating step.

The auxiliary losses provide **deep supervision** — each expert is
independently trained to classify subtypes at its own scale.  The
`Loss_cell` term only kicks in once `cell_warmup_start` epochs have
elapsed (see Cell Warmup Curriculum in Scale Dropout section below).

Each per-expert auxiliary loss is logged separately to W&B/TensorBoard
for monitoring.

**Validation & checkpointing:**

- Validation runs every 5 epochs.
- Tracked metrics: `val_loss`, `val_accuracy`, `val_F1`, `val_AUROC`.
- Early stopping: patience = 20 epochs, `min_delta = 0.001`.
- Best model checkpoint saved by minimum `val_loss`.
- Metrics are stored inside the checkpoint for offline inspection.

---

### Model Types

| `--model_type` | Class | Description |
|---|---|---|
| `attention` | `AttentionMIL` | Standard gated-attention MIL (bag-level attention → classify) |
| `additive` | `AdditiveMIL` | Per-class attention + per-instance classifiers → additive pooling |
| `hierarchical` | `HierarchicalMIL` | Three-expert hierarchical MIL (region + patch + cell) with gated fusion |

---

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url> && cd predictive_model

# 2. Create / activate conda environment (recommended)
mamba create -n ibd_mil python=3.10
mamba activate ibd_mil

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project in editable mode (optional)
pip install -e .
```

Key dependencies: `torch`, `lightning`, `lazyslide`, `wsidata`,
`scanpy`, `geopandas`, `shapely`, `h5py`, `scikit-learn`.

---

## Running the Pipeline

### 1. Feature Extraction (SLURM array job)

```bash
cd sbatchers
sbatch run_feature_extraction.sh \
    --array ../data/metadata/slides.csv \
    --config ../configs/feature_extraction.yaml
```

Each array task processes one slide end-to-end:
segmentation → tiling → H5 metadata → Virchow2 extraction → save embeddings.

### 2. Create Cross-Validation Folds

```bash
python scripts/create_folds.py
```

### 3. Train (SLURM array over 5 folds)

```bash
cd sbatchers
sbatch run_MIL_training.sh
```
---
