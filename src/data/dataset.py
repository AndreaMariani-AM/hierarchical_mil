
from pathlib import Path
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class RepresentationsDataset(Dataset):
    """
    Dataset for whole slide image (WSI) embeddings for Multiple Instance Learning.
    
    Assumes embeddings are stored as .h5ad files with shape (n_tiles, embedding_dim).
    """
    
    def __init__(
        self,
        csv_path: str,
        representation_dir: str,
        max_tiles: Optional[int] = None,
        split: str = 'train',
    ):
        """
        Args:
            csv_path: Path to CSV file containing slide metadata
            representation_dir: Directory where embeddings are stored
            max_tiles: Maximum number of tiles to use (random sampling if exceeded)
            split: 'train' or 'val' to indicate dataset split
        """
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['use_slide'] == True].reset_index(drop=True)
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        self.max_tiles = max_tiles
        self.representation_dir = representation_dir
      
        print(f"Loaded {len(self.metadata)} slides for split '{split}'")
        
        # Get list of slide IDs that have both embeddings and labels
        self.label_dict = self._get_label_dict()
        self.slide_ids = list(self.label_dict.keys())
        
        if len(self.slide_ids) == 0:
            raise ValueError(f"No slides found in csv file: {csv_path} for split: {split}")
    
    def _get_label_dict(self) -> Dict[str, int]:
        """Create a dictionary mapping slide IDs to integer labels."""
        sampels_list = self.metadata['Slide'].apply(lambda x: os.path.basename(x).split('.')[0]).tolist()
        conditions_list = self.metadata['Condition'].tolist()
        le = LabelEncoder()
        encoded_labels = torch.tensor(le.fit_transform(conditions_list), dtype=torch.int64)
        return {slide: label for slide, label in zip(sampels_list, encoded_labels)}
    
    def _load_embeddings(self, slide_id: str) -> torch.Tensor:
        """
        Load embeddings from disk, auto-detecting the file format.

        Supports two layouts:

        * **Hierarchical H5** — raw HDF5 with ``/mid/H`` dataset of
          shape ``(N, D)``.  Only the embeddings are read; region and
          coordinate metadata are ignored.
        * **AnnData H5** (legacy) — ``anndata``-formatted ``.h5`` file
          where ``adata.X`` holds the ``(N, D)`` embeddings.

        The format is detected by probing for the ``mid/H`` key.

        Returns
        -------
        embeddings : Tensor ``(N, D)`` float32
        """
        file_path = os.path.join(self.representation_dir, f"{slide_id}.h5")

        # Probe file to decide format
        with h5py.File(file_path, "r") as f:
            is_hierarchical = "mid" in f
            if is_hierarchical:
                embeddings = torch.from_numpy(
                    np.array(f["mid/H_patch"], dtype=np.float32)
                ) #using the full CLS+mean embeddings.

            else:
                adata = ad.read_h5ad(file_path)
                embeddings = torch.from_numpy(adata.X).float()

        # Ensure float32
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()

        return embeddings
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """4
        Returns:
            embeddings: Tensor of shape (n_tiles, embedding_dim)
            label: Integer label
            slide_id: String identifier for the slide
        """
        slide_id = self.slide_ids[idx]
        label = self.label_dict[slide_id]
        
        # Load embeddings
        embeddings = self._load_embeddings(slide_id)
        
        # Sample tiles if max_tiles is set and we have more tiles
        # DEFINE A SAMPLING STRATEGY HERE
        if self.max_tiles is not None and embeddings.shape[0] > self.max_tiles:
            pass
            # indices = torch.randperm(embeddings.shape[0])[:self.max_tiles]
            # embeddings = embeddings[indices]
        
        return (embeddings, label, slide_id)


# ======================================================================
#  Hierarchical MIL dataset  +  custom collate
# ======================================================================


def hierarchical_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate for ``HierarchicalRepresentationsDataset``.

    Designed for **batch_size = 1** (standard for MIL) and simply unpacks
    the single sample, adding a leading batch dim to tensors where appropriate.

    Expected input::

        batch = [
            (h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)
        ]
    """
    assert len(batch) == 1, (
        f"hierarchical_collate_fn requires batch_size=1, got {len(batch)} samples"
    )
    h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions = batch[0]

    # Add batch dim to tensor inputs (keeps parity with default collate)
    h_patches = h_patches.unsqueeze(0)  # (1, N, 1280)
    h_region  = h_region.unsqueeze(0)   # (1, N, 2560)
    h_cells   = h_cells.unsqueeze(0)    # (1, N, 256, 1280)

    # Label → (1,) tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, dtype=torch.int64)
    label = label.unsqueeze(0)

    # slide_id stays as a plain string
    # region_ids: keep as (N,) — model squeezes batch dim on embedding tensors
    # n_regions: plain int

    return (h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)


class HierarchicalRepresentationsDataset(Dataset):
    """
    Dataset for hierarchical MIL training.

    Loads **patch embeddings** and **region assignments** from the
    hierarchical H5 files produced by
    :func:`~src.data.preprocessing.save_tile_region_h5` /
    :func:`~src.data.preprocessing.save_embeddings_to_h5`.

    Optionally provides a ``fetch_patches_fn`` callable that reads
    raw 256×256 patch images from the WSI on-the-fly (for the
    CellExpert's frozen dense backbone).

    H5 layout (per slide)
    ---------------------
    ::

        /mid/tile_id    : (N,)            int32
        /mid/region_id  : (N,)            int32   patch→region map
        /mid/H_patch    : (N, 1280)       float16  CLS token — PatchExpert input
        /mid/H_region   : (N, 2560)       float16  CLS+mean — RegionExpert input
        /cells/H        : (N, 256, 1280)  float16  dense tokens — CellExpert input
        /regions/region_id : (R,)         int32
        /regions/xy_2048   : (R, 4)       int32

    ``__getitem__`` returns
    -----------------------
    ::

        (h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)

    Use :func:`hierarchical_collate_fn` as the DataLoader's ``collate_fn``.
    """

    def __init__(
        self,
        csv_path: str,
        h5_dir: str,
        split: str = "train",
        max_tiles: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to fold CSV produced by ``create_folds.py``.
            Required columns: ``Slide``, ``Condition``, ``split``.
        h5_dir : str
            Directory containing ``{slide_name}.h5`` hierarchical files
            (e.g. ``data/features/extracted_features/virchow2_256_hierarchical``).
        split : str
            ``'train'`` or ``'val'``.
        max_tiles : int or None
            If set, randomly sub-sample patches (and update region_ids)
            when a slide exceeds this count.
        """
        self.h5_dir = h5_dir
        self.max_tiles = max_tiles

        # ── Load & filter metadata ───────────────────────────────────
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['use_slide'] == True].reset_index(drop=True) #only for testing
        self.metadata = self.metadata[
            self.metadata["split"] == split
        ].reset_index(drop=True)

        # ── Build label dict (slide_name → int) ─────────────────────
        self.label_dict = self._build_lookups()
        self.slide_ids = list(self.label_dict.keys())

        if len(self.slide_ids) == 0:
            raise ValueError(
                f"No slides found in {csv_path} for split '{split}'"
            )
        print(
            f"[HierarchicalRepresentationsDataset] "
            f"Loaded {len(self.slide_ids)} slides for split '{split}'"
        )

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _build_lookups(self) -> Dict[str, int]:
        """Return *label_dict* from metadata CSV."""
        slide_names = (
            self.metadata["Slide"]
            .apply(lambda x: os.path.basename(x).split(".")[0])
            .tolist()
        )
        conditions = self.metadata["Condition"].tolist()

        le = LabelEncoder()
        encoded = torch.tensor(
            le.fit_transform(conditions), dtype=torch.int64
        )

        return {name: lab for name, lab in zip(slide_names, encoded)}

    def _h5_path(self, slide_id: str) -> str:
        return os.path.join(self.h5_dir, f"{slide_id}.h5")

    def _load_h5(
        self, slide_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Read the hierarchical H5 file for one slide.

        Returns
        -------
        h_patches  : Tensor ``(N, 1280)``       float32  — CLS token per patch
        h_region   : Tensor ``(N, 2560)``       float32  — CLS+mean per patch
        h_cells    : Tensor ``(N, 256, 1280)``  float32  — dense tokens per patch
        region_ids : Tensor ``(N,)``            int64
        n_regions  : int
        """
        h5_path = self._h5_path(slide_id)
        with h5py.File(h5_path, "r") as f:
            h_patches = torch.from_numpy(
                np.array(f["mid/H_patch"], dtype=np.float32)
            )
            h_region = torch.from_numpy(
                np.array(f["mid/H_region"], dtype=np.float32)
            )
            h_cells = torch.from_numpy(
                np.array(f["cells/H"], dtype=np.float32)
            )
            region_ids = torch.from_numpy(
                np.array(f["mid/region_id"], dtype=np.int64)
            )
            n_regions = f["regions/region_id"].shape[0]

        return h_patches, h_region, h_cells, region_ids, n_regions

        # def _get_slide_mpp(self, slide_id: str) -> float:
        #     """
        #     Retrieve the base-level MPP for a slide.

        #     Currently returns a sensible default (0.5).  Override or extend
        #     this if your metadata CSV contains per-slide MPP values.
        #     """
        #     # TODO: read from metadata if available, e.g.:
        #     #   return self.metadata.loc[
        #     #       self.metadata['Slide'].str.contains(slide_id), 'mpp'
        #     #   ].values[0]
        #     return 0.5

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, torch.Tensor, int]:
        """
        Returns
        -------
        h_patches  : Tensor ``(N, 1280)``      — CLS per patch
        h_region   : Tensor ``(N, 2560)``      — CLS+mean per patch
        h_cells    : Tensor ``(N, 256, 1280)`` — dense tokens per patch
        label      : int
        slide_id   : str
        region_ids : Tensor ``(N,)``
        n_regions  : int
        """
        slide_id = self.slide_ids[idx]
        label = self.label_dict[slide_id]

        h_patches, h_region, h_cells, region_ids, n_regions = self._load_h5(slide_id)

        # ── Optional random sub-sampling ─────────────────────────────
        if (
            self.max_tiles is not None
            and h_patches.shape[0] > self.max_tiles
        ):
            perm = torch.randperm(h_patches.shape[0])[: self.max_tiles]
            perm, _ = perm.sort()  # keep spatial ordering
            h_patches  = h_patches[perm]
            h_region   = h_region[perm]
            h_cells    = h_cells[perm]
            region_ids = region_ids[perm]

        return (h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)


# ======================================================================
#  Patient-level MIL dataset  +  custom collate
# ======================================================================

def patient_hierarchical_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate for ``PatientHierarchicalDataset``.

    Designed for **batch_size = 1**.  The ``slides_list`` contains raw CPU
    tensors (no unsqueeze); device transfer and unsqueeze happen inside
    ``MILTrainer._process_patient_batch``.

    Expected input::

        batch = [
            (slides_list, label, patient_id)
        ]

    where each element of ``slides_list`` is::

        (h_patches (N,D_p), h_region (N,D_r), h_cells (N,256,D_c),
         region_ids (N,), n_regions int)

    Returns
    -------
    (slides_list, label, patient_id)
        label : Tensor ``(1,)`` int64
    """
    assert len(batch) == 1, (
        f"patient_hierarchical_collate_fn requires batch_size=1, got {len(batch)} samples"
    )
    slides_list, label, patient_id = batch[0]

    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, dtype=torch.int64)
    label = label.unsqueeze(0)  # (1,)

    return (slides_list, label, patient_id)


class PatientHierarchicalDataset(HierarchicalRepresentationsDataset):
    """
    Patient-level dataset for hierarchical MIL training.

    Groups slides by ``patient_id`` and returns all slides for a patient
    in a single sample.  The per-patient label is the shared ``Condition``
    label for all slides belonging to that patient (consistency is
    asserted on construction).

    Does **not** call ``super().__init__()`` — the parent builds
    slide-level lookups that are unused here.  Only
    ``_load_h5`` and ``_h5_path`` are inherited (both depend only on
    ``self.h5_dir``).

    ``__getitem__`` returns
    -----------------------
    ::

        (slides_list, patient_label, patient_id)

    where ``slides_list`` is a Python list of tuples::

        (h_patches (N,D_p), h_region (N,D_r), h_cells (N,256,D_c),
         region_ids (N,), n_regions int)

    Use :func:`patient_hierarchical_collate_fn` as the DataLoader's
    ``collate_fn``.
    """

    def __init__(
        self,
        csv_path: str,
        h5_dir: str,
        split: str = "train",
        max_tiles: Optional[int] = None,
    ):
        # Do NOT call super().__init__() — it builds slide-level lookups
        # that are not needed here and would duplicate the metadata filtering.
        # Set only the shared attributes that _load_h5 / _h5_path require.
        self.h5_dir = h5_dir
        self.max_tiles = max_tiles

        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata["use_slide"] == True].reset_index(drop=True)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

        self._build_patient_lookups()

        if len(self.patient_list) == 0:
            raise ValueError(
                f"No patients found in {csv_path} for split '{split}'"
            )
        print(
            f"[PatientHierarchicalDataset] "
            f"Loaded {len(self.patient_list)} patients "
            f"({len(self.metadata)} slides) for split '{split}'"
        )

    def _build_patient_lookups(self) -> None:
        """Build patient-level index structures from filtered metadata."""
        slide_names = (
            self.metadata["Slide"]
            .apply(lambda x: os.path.basename(x).split(".")[0])
            .tolist()
        )
        conditions = self.metadata["Condition"].tolist()
        patient_ids = self.metadata["patient_id"].tolist()

        le = LabelEncoder()
        encoded = le.fit_transform(conditions)  # int array

        patient_to_slides: Dict[str, List[str]] = {}
        patient_to_label: Dict[str, int] = {}

        for slide_name, pid, label_int in zip(slide_names, patient_ids, encoded):
            if pid not in patient_to_slides:
                patient_to_slides[pid] = []
                patient_to_label[pid] = int(label_int)
            else:
                assert patient_to_label[pid] == int(label_int), (
                    f"Patient {pid} has slides with conflicting labels: "
                    f"{le.classes_[patient_to_label[pid]]} vs {le.classes_[int(label_int)]}"
                )
            patient_to_slides[pid].append(slide_name)

        self.patient_list: List[str] = sorted(patient_to_slides.keys())
        self.patient_to_slides: Dict[str, List[str]] = patient_to_slides
        self.patient_label_dict: Dict[str, torch.Tensor] = {
            pid: torch.tensor(lbl, dtype=torch.int64)
            for pid, lbl in patient_to_label.items()
        }

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[Tuple], torch.Tensor, str]:
        """
        Returns
        -------
        slides_list   : list of (h_patches, h_region, h_cells, region_ids, n_regions, fetch_fn)
        patient_label : Tensor ``()`` int64
        patient_id    : str
        """
        patient_id = self.patient_list[idx]
        patient_label = self.patient_label_dict[patient_id]

        slides_list = []
        for slide_name in self.patient_to_slides[patient_id]:
            h_patches, h_region, h_cells, region_ids, n_regions = self._load_h5(slide_name)

            # Optional random sub-sampling (same logic as parent __getitem__)
            if self.max_tiles is not None and h_patches.shape[0] > self.max_tiles:
                perm = torch.randperm(h_patches.shape[0])[: self.max_tiles]
                perm, _ = perm.sort()
                h_patches  = h_patches[perm]
                h_region   = h_region[perm]
                h_cells    = h_cells[perm]
                region_ids = region_ids[perm]

            slides_list.append(
                (h_patches, h_region, h_cells, region_ids, n_regions)
            )

        return (slides_list, patient_label, patient_id)


class SeqDataset(Dataset):
    """
    Custom Dataset for loading sequences of features (e.g. [64 x 384] tensors) for DINO training stage 2.
    Assumes data is stored as .pt files containing tensors of shape (seq_len, feature_dim).
    """
    def __init__(self, dataroot: str, transform: Optional[Callable] = None):
        self.dataroot = dataroot
        self.transform = transform
        self.file_paths = list(Path(dataroot).glob("*.pt"))
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        features = torch.load(file_path)  # shape (seq_len, feature_dim)
        label = torch.zeros(1,1)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label