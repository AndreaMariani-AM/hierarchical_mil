"""
Spatial construction and visualization for MIL attention/contribution scores.

Pipeline
--------
1. :func:`read_h5_spatial_metadata`  — load bboxes + region map from H5.
2. :func:`build_patch_geodataframe`  — create a GeoDataFrame of patches
   enriched with per-patch attention + contribution scores.
3. :func:`build_wsidata_with_predictions`  — attach those GeoDataFrames
   to an opened WSIData object for integration with lazyslide tools.
4. Static matplotlib plots  — :func:`plot_patch_scores`,
   :func:`plot_region_scores`, :func:`plot_expert_comparison`,
   :func:`plot_cell_tokens`.
5. Interactive napari viewer  — :func:`launch_napari_viewer`.

Coordinate system
-----------------
All bounding boxes (``xy_256``, ``xy_2048``) stored in the hierarchical
H5 are in **native slide pixel coordinates** (base resolution) with
format ``[xmin, ymin, xmax, ymax]``.  Thumbnail rendering scales these
by ``native_mpp / thumbnail_mpp`` so coordinates stay consistent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes
from shapely.geometry import box as shapely_box

# ── project imports ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Re-export unpack_attn_dict here so callers can import from either module
from src.evaluation.inference_utils import unpack_attn_dict, get_top_k_patch_indices  # noqa: F401


# ======================================================================
# 0.  Score utilities
# ======================================================================

def normalize_scores(
    scores: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize an array of scores to [0, 1].

    Parameters
    ----------
    scores : np.ndarray
        1-D array of raw scores (attention weights or contributions).
    method : str
        ``"minmax"`` — linear rescale to [0, 1] (default).
        ``"rank"``   — percentile-rank normalization (robust to outliers).
        ``"relative"`` — normalize by the sum of absolute values.
        ``"none"``   — return scores unchanged.

    Returns
    -------
    np.ndarray
        Normalized scores with the same shape as *scores*.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if method == "none":
        return scores
    if method == "rank":
        from scipy.stats import rankdata
        return rankdata(scores) / len(scores)
    if method == 'relative':
        return np.abs(scores) / (np.abs(scores).sum())
    # minmax (default)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ======================================================================
# 1.  H5 metadata reader
# ======================================================================

def read_h5_spatial_metadata(h5_path: Union[str, Path]) -> Dict:
    """Read bounding boxes, region assignments, and MPP from an H5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to a hierarchical H5 file produced by
        :func:`src.data.preprocessing.save_tile_region_h5`.

    Returns
    -------
    dict with keys:
        ``xy_256``     — np.ndarray ``(N, 4)`` int32 — patch bboxes
                         ``[xmin, ymin, xmax, ymax]`` in native pixels.
        ``xy_2048``    — np.ndarray ``(R, 4)`` int32 — region bboxes.
        ``region_ids`` — np.ndarray ``(N,)`` int32 — patch → region map.
        ``n_regions``  — int.
        ``mpp``        — float, native tiling resolution (microns/pixel).
    """
    with h5py.File(h5_path, "r") as f:
        xy_256 = f["mid/xy_256"][:].astype(np.int32)
        xy_2048 = f["regions/xy_2048"][:].astype(np.int32)
        region_ids = f["mid/region_id"][:].astype(np.int32)
        n_regions = int(f["regions/region_id"].shape[0])
        mpp_raw = f["slide_metadata/mpp"][()]
        mpp = float(np.asarray(mpp_raw).flat[0])
    return {
        "xy_256": xy_256,
        "xy_2048": xy_2048,
        "region_ids": region_ids,
        "n_regions": n_regions,
        "mpp": mpp,
    }


# ======================================================================
# 2.  Segmentation mask loader
# ======================================================================

def load_segmentation_mask(seg_mask_path: Union[str, Path]) -> List:
    """Load tissue segmentation polygons from an H5 mask file.

    Parameters
    ----------
    seg_mask_path : str or Path
        Path to ``{slide_name}_segmentation_mask.h5`` produced during
        the segmentation step in the feature extraction pipeline.

    Returns
    -------
    list of shapely geometries
        Tissue polygon geometries in native slide pixel coordinates.

    Raises
    ------
    FileNotFoundError
        If *seg_mask_path* does not exist.
    KeyError
        If no recognisable WKT dataset is found in the H5 file.
    """
    from shapely import wkt as shapely_wkt

    seg_mask_path = Path(seg_mask_path)
    if not seg_mask_path.exists():
        raise FileNotFoundError(f"Segmentation mask not found: {seg_mask_path}")

    with h5py.File(seg_mask_path, "r") as f:
        # Try common dataset names written by feature_extraction.py
        candidate_keys = ["wkt", "geometries", "tissue_geometries", "tissue", "tissues"]
        wkt_key = None
        for k in candidate_keys:
            if k in f:
                wkt_key = k
                break
        if wkt_key is None:
            # Fall back: use the first top-level dataset
            all_keys = list(f.keys())
            if not all_keys:
                raise KeyError(f"No datasets found in {seg_mask_path}")
            wkt_key = all_keys[0]

        raw = f[wkt_key][:]
        # Stored as bytes or str
        wkt_strings = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw
        ]

    return [shapely_wkt.loads(s) for s in wkt_strings]


# ======================================================================
# 3.  GeoDataFrame construction
# ======================================================================

def build_patch_geodataframe(
    attn_dict: Dict,
    xy_256: np.ndarray,
    region_ids: Optional[np.ndarray] = None,
    normalize: str = "minmax",
) -> gpd.GeoDataFrame:
    """Build a patch-level GeoDataFrame enriched with attention/contribution scores.

    Works for both model types.  For hierarchical models the *attn_dict*
    must have been produced by :func:`~src.evaluation.inference_utils.unpack_attn_dict`
    (i.e., expert outputs already converted to plain dicts).  For additive
    models only the ``'patch'`` key is required.

    Score columns added
    -------------------
    **Always present** (any model):
        ``patch_attn``         — mean attention across classes, normalized.
        ``patch_contribution`` — mean |contribution| across classes, normalized.

    **Hierarchical only** (when ``'region'`` key exists in *attn_dict*):
        ``region_attn``        — attention of the owning region, normalized.
        ``region_contribution``— contribution of the owning region, normalized.
        ``combined_score``     — ``patch_contribution × region_contribution``,
                                  normalized to [0, 1].

    Parameters
    ----------
    attn_dict : dict
        Normalised attention dict from :func:`~src.evaluation.inference_utils.unpack_attn_dict`.
    xy_256 : np.ndarray, shape ``(N, 4)``
        Patch bounding boxes ``[xmin, ymin, xmax, ymax]`` in native pixels.
    region_ids : np.ndarray, shape ``(N,)``, optional
        Patch → region index map.  Required only for hierarchical models to
        propagate region scores down to the patch level.
    normalize : str
        Score normalization method passed to :func:`normalize_scores`.

    Returns
    -------
    GeoDataFrame
        One row per patch.  ``geometry`` column holds a shapely ``box``.
    """
    N = len(xy_256)

    # ── Patch scores ─────────────────────────────────────────────────
    patch_attn_raw = attn_dict["patch"]["attn_weights"]      # Tensor (N, C) or ndarray
    patch_contrib_raw = attn_dict["patch"]["contributions"]  # Tensor (N, C) or ndarray

    if isinstance(patch_attn_raw, torch.Tensor):
        patch_attn_raw = patch_attn_raw.numpy()
    if isinstance(patch_contrib_raw, torch.Tensor):
        patch_contrib_raw = patch_contrib_raw.numpy()

    patch_attn = normalize_scores(patch_attn_raw.mean(axis=-1), method=normalize)
    patch_contrib = normalize_scores(np.abs(patch_contrib_raw).mean(axis=-1), method=normalize)

    data = {
        "geometry": [shapely_box(*xy) for xy in xy_256],
        "patch_attn": patch_attn,
        "patch_contribution": patch_contrib,
    }

    # ── Region scores propagated to patches ──────────────────────────
    if "region" in attn_dict and region_ids is not None:
        reg = attn_dict["region"]
        region_attn_raw = reg["attn_weights"]     # (R, C)
        region_contrib_raw = reg["contributions"] # (R, C)

        if isinstance(region_attn_raw, torch.Tensor):
            region_attn_raw = region_attn_raw.numpy()
        if isinstance(region_contrib_raw, torch.Tensor):
            region_contrib_raw = region_contrib_raw.numpy()

        region_attn_per_region = normalize_scores(
            region_attn_raw.mean(axis=-1), method=normalize
        )
        region_contrib_per_region = normalize_scores(
            np.abs(region_contrib_raw).mean(axis=-1), method=normalize
        )

        region_ids_np = (
            region_ids.numpy() if isinstance(region_ids, torch.Tensor) else np.asarray(region_ids)
        )
        # Clamp indices to valid range (safety guard)
        region_ids_np = np.clip(region_ids_np, 0, len(region_attn_per_region) - 1)

        data["region_attn"] = region_attn_per_region[region_ids_np]
        data["region_contribution"] = region_contrib_per_region[region_ids_np]
        data["combined_score"] = normalize_scores(
            data["patch_contribution"] * data["region_contribution"], method=normalize
        )

    gdf = gpd.GeoDataFrame(data, geometry="geometry")
    return gdf


def build_region_geodataframe(
    attn_dict: Dict,
    xy_2048: np.ndarray,
    normalize: str = "minmax",
) -> gpd.GeoDataFrame:
    """Build a region-level GeoDataFrame enriched with attention/contribution scores.

    Parameters
    ----------
    attn_dict : dict
        Normalised attention dict.  Must contain a ``'region'`` key
        (hierarchical models only).
    xy_2048 : np.ndarray, shape ``(R, 4)``
        Region bounding boxes ``[xmin, ymin, xmax, ymax]`` in native pixels.
    normalize : str
        Score normalization method.

    Returns
    -------
    GeoDataFrame
        One row per 2048×2048 region with ``region_attn`` and
        ``region_contribution`` columns.

    Raises
    ------
    KeyError
        If *attn_dict* has no ``'region'`` key (additive model output).
    """
    if "region" not in attn_dict:
        raise KeyError(
            "'region' key not found in attn_dict.  "
            "build_region_geodataframe requires a hierarchical model output."
        )

    reg = attn_dict["region"]
    region_attn_raw = reg["attn_weights"]     # (R, C)
    region_contrib_raw = reg["contributions"] # (R, C)

    if isinstance(region_attn_raw, torch.Tensor):
        region_attn_raw = region_attn_raw.numpy()
    if isinstance(region_contrib_raw, torch.Tensor):
        region_contrib_raw = region_contrib_raw.numpy()

    return gpd.GeoDataFrame(
        {
            "geometry": [shapely_box(*xy) for xy in xy_2048],
            "region_attn": normalize_scores(region_attn_raw.mean(axis=-1), method=normalize),
            "region_contribution": normalize_scores(
                np.abs(region_contrib_raw).mean(axis=-1), method=normalize
            ),
        },
        geometry="geometry",
    )


# ======================================================================
# 4.  WSIData integration
# ======================================================================

def build_wsidata_with_predictions(
    wsi_path: Union[str, Path],
    h5_path: Union[str, Path],
    attn_dict: Dict,
    model_type: str = "hierarchical",
    seg_mask_path: Optional[Union[str, Path]] = None,
    normalize: str = "minmax",
):
    """Open a WSIData object and attach prediction GeoDataFrames to it.

    After calling this function the returned WSIData has:
        * ``wsi.shapes['tiles']``        — patch GeoDataFrame with scores.
        * ``wsi.shapes['region_tiles']`` — region GeoDataFrame (hierarchical
          only).
        * ``wsi.shapes['tissues']``      — segmentation polygons (if
          *seg_mask_path* provided).

    Parameters
    ----------
    wsi_path : str or Path
        Path to the raw WSI file (e.g. ``.svs``, ``.ndpi``).
    h5_path : str or Path
        Path to the hierarchical H5 file for this slide.
    attn_dict : dict
        Normalised attention dict from :func:`~src.evaluation.inference_utils.unpack_attn_dict`.
    model_type : str
        ``"hierarchical"`` or ``"additive"`` / ``"attention"``.
    seg_mask_path : str or Path, optional
        Path to the segmentation mask H5 file.  When provided, tissue
        polygons are loaded and attached to the WSIData object.
    normalize : str
        Score normalization method for all score columns.

    Returns
    -------
    tuple ``(wsi, gdfs)``
        *wsi*  : opened ``WSIData`` with prediction shapes attached.
        *gdfs* : dict with keys ``'patches'`` and (hierarchical only)
                 ``'regions'`` — the raw GeoDataFrames for independent use.

    Notes
    -----
    WSIData shape attachment uses ``wsi.shapes[key] = ShapesModel.parse(gdf)``.
    If your version of *wsidata* / *spatialdata* uses a different API
    (e.g. ``wsi.add_shapes``), adapt the assignment lines accordingly.
    """
    from wsidata import open_wsi
    from spatialdata.models import ShapesModel

    meta = read_h5_spatial_metadata(h5_path)

    patches_gdf = build_patch_geodataframe(
        attn_dict,
        meta["xy_256"],
        region_ids=meta["region_ids"],
        normalize=normalize,
    )

    gdfs = {"patches": patches_gdf}

    wsi = open_wsi(str(wsi_path))

    # Attach patches
    try:
        wsi.shapes["tiles"] = ShapesModel.parse(patches_gdf)
    except Exception:
        # Fallback: direct assignment (older spatialdata / wsidata versions)
        wsi.shapes["tiles"] = patches_gdf

    # Attach regions (hierarchical only)
    if model_type == "hierarchical" and "region" in attn_dict:
        regions_gdf = build_region_geodataframe(
            attn_dict, meta["xy_2048"], normalize=normalize
        )
        gdfs["regions"] = regions_gdf
        try:
            wsi.shapes["region_tiles"] = ShapesModel.parse(regions_gdf)
        except Exception:
            wsi.shapes["region_tiles"] = regions_gdf

    # Attach segmentation mask
    if seg_mask_path is not None:
        tissue_geoms = load_segmentation_mask(seg_mask_path)
        tissue_gdf = gpd.GeoDataFrame(
            {"geometry": tissue_geoms},
            geometry="geometry",
        )
        try:
            wsi.shapes["tissues"] = ShapesModel.parse(tissue_gdf)
        except Exception:
            wsi.shapes["tissues"] = tissue_gdf

    return wsi, gdfs


# ======================================================================
# 5.  Thumbnail helper
# ======================================================================

def get_slide_thumbnail(
    wsi_path: Union[str, Path],
    mpp: float = 8.0,
    native_mpp: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return a low-resolution thumbnail of the slide.

    Parameters
    ----------
    wsi_path : str or Path
        Path to the raw WSI file.
    mpp : float
        Desired thumbnail resolution in microns per pixel (default 8.0).
    native_mpp : float, optional
        Known native mpp of the slide.  If ``None``, the function tries
        to infer it from the WSI metadata.

    Returns
    -------
    tuple ``(thumbnail, scale)``
        *thumbnail* : np.ndarray ``(H, W, 3)`` uint8 RGB image.
        *scale*     : float — multiply native-pixel coordinates by this to
                      get thumbnail-pixel coordinates.

    Notes
    -----
    Attempts ``wsi.get_thumbnail`` first (wsidata API), then falls back to
    openslide ``get_thumbnail``.
    """
    try:
        from wsidata import open_wsi
        wsi = open_wsi(str(wsi_path))
        # Always read native MPP from WSI API — overrides any caller-supplied
        # value so wrong defaults (e.g. tiling MPP 0.5 vs scanner MPP 0.25)
        # cannot silently misalign overlays.
        try:
            native_mpp = float(wsi.properties.get("mpp", 0.5))
        except Exception:
            native_mpp = 0.5
        scale = native_mpp / mpp
        width = int(wsi.properties["width"] * scale)
        height = int(wsi.properties["height"] * scale)
        try:
            thumbnail = np.asarray(wsi.get_thumbnail((width, height)))
        except AttributeError:
            # wsidata may expose the reader directly
            thumbnail = np.asarray(wsi.reader.get_thumbnail((width, height)))
        return thumbnail, scale
    except Exception:
        pass

    # Fallback: openslide
    try:
        import openslide
        slide = openslide.open_slide(str(wsi_path))
        # Always read native MPP from WSI API (see wsidata path above).
        native_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
        scale = native_mpp / mpp
        w_full = int(slide.dimensions[0])
        h_full = int(slide.dimensions[1])
        width = max(1, int(w_full * scale))
        height = max(1, int(h_full * scale))
        thumbnail = np.asarray(slide.get_thumbnail((width, height)))
        return thumbnail, scale
    except ImportError as e:
        raise ImportError(
            "Could not open WSI file.  Install openslide-python or wsidata."
        ) from e


# ======================================================================
# 6.  Static matplotlib plots
# ======================================================================

def _get_slide_extent(xy_256: np.ndarray) -> Tuple[float, float]:
    """Infer slide extent in native pixels from patch bboxes."""
    x_max = int(xy_256[:, 2].max())
    y_max = int(xy_256[:, 3].max())
    return float(x_max), float(y_max)


def plot_patch_scores(
    patches_gdf: gpd.GeoDataFrame,
    score_col: str = "patch_contribution",
    wsi_path: Optional[Union[str, Path]] = None,
    native_mpp: Optional[float] = None,
    thumbnail_mpp: float = 8.0,
    cmap: str = "RdBu_r",
    alpha: float = 0.55,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-patch scores on a slide thumbnail (or blank canvas).

    Parameters
    ----------
    patches_gdf : GeoDataFrame
        Output of :func:`build_patch_geodataframe`.
    score_col : str
        Column name to visualize.  Must be a float column in *patches_gdf*.
    wsi_path : str or Path, optional
        Path to the raw WSI file.  When provided, the actual H&E image is
        used as background.  When ``None`` patches are drawn on a white
        canvas sized from the bounding boxes.
    native_mpp : float, optional
        Native tiling mpp (read from H5 when available).  Used to compute
        the scale factor for the thumbnail background.
    thumbnail_mpp : float
        MPP at which the background thumbnail is rendered.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Opacity of the colored rectangles overlaid on the thumbnail.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Figure title.  Defaults to *score_col*.
    save_path : str or Path, optional
        If given, the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    scores = patches_gdf[score_col].values.astype(np.float64)
    cm = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())

    fig, ax = plt.subplots(figsize=figsize)

    # Background: H&E thumbnail or white canvas
    if wsi_path is not None:
        thumbnail, scale = get_slide_thumbnail(
            wsi_path, mpp=thumbnail_mpp, native_mpp=native_mpp
        )
        ax.imshow(thumbnail, origin="upper")
    else:
        # Infer canvas size from bboxes
        bounds = patches_gdf.total_bounds  # (xmin, ymin, xmax, ymax)
        native_mpp_eff = native_mpp or 0.5
        scale = native_mpp_eff / thumbnail_mpp
        thumb_w = max(1, int(bounds[2] * scale))
        thumb_h = max(1, int(bounds[3] * scale))
        ax.set_xlim(0, thumb_w)
        ax.set_ylim(thumb_h, 0)  # invert y: image convention

    # Overlay patch rectangles
    for row, score in zip(patches_gdf.geometry, scores):
        x0, y0, x1, y1 = row.bounds
        rect = mpatches.Rectangle(
            (x0 * scale, y0 * scale),
            (x1 - x0) * scale,
            (y1 - y0) * scale,
            linewidth=0,
            facecolor=cm(norm(score)),
            alpha=alpha,
        )
        ax.add_patch(rect)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label=score_col)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or score_col, fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_region_scores(
    regions_gdf: gpd.GeoDataFrame,
    score_col: str = "region_contribution",
    wsi_path: Optional[Union[str, Path]] = None,
    native_mpp: Optional[float] = None,
    thumbnail_mpp: float = 8.0,
    cmap: str = "RdBu_r",
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-region scores on a slide thumbnail.

    Identical signature to :func:`plot_patch_scores` but operates on the
    coarser 2048×2048 region GeoDataFrame.
    """
    return plot_patch_scores(
        patches_gdf=regions_gdf,
        score_col=score_col,
        wsi_path=wsi_path,
        native_mpp=native_mpp,
        thumbnail_mpp=thumbnail_mpp,
        cmap=cmap,
        alpha=alpha,
        figsize=figsize,
        title=title or score_col,
        save_path=save_path,
    )


def plot_expert_comparison(
    patches_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame,
    wsi_path: Optional[Union[str, Path]] = None,
    native_mpp: Optional[float] = None,
    thumbnail_mpp: float = 8.0,
    cmap: str = "RdBu_r",
    alpha: float = 0.55,
    figsize: Tuple[float, float] = (18, 8),
    slide_id: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Three-panel comparison: patch contributions | region contributions | combined.

    Only for hierarchical models.  Raises ``KeyError`` if *patches_gdf*
    lacks a ``'combined_score'`` column.

    Parameters
    ----------
    patches_gdf : GeoDataFrame
        Must contain ``patch_contribution``, ``region_contribution``, and
        ``combined_score`` columns.
    regions_gdf : GeoDataFrame
        Must contain ``region_contribution``.
    wsi_path, native_mpp, thumbnail_mpp, cmap, alpha, figsize
        Same as :func:`plot_patch_scores`.
    slide_id : str, optional
        Used as the figure suptitle.
    save_path : str or Path, optional
        If given, the figure is saved before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    for col in ("patch_contribution", "combined_score"):
        if col not in patches_gdf.columns:
            raise KeyError(
                f"Column '{col}' not found in patches_gdf.  "
                "plot_expert_comparison requires a hierarchical model output with "
                "both patch and region scores."
            )

    thumbnail = None
    scale = 1.0
    if wsi_path is not None:
        thumbnail, scale = get_slide_thumbnail(
            wsi_path, mpp=thumbnail_mpp, native_mpp=native_mpp
        )

    panels = [
        (patches_gdf, "patch_contribution", "Patch contributions"),
        (regions_gdf, "region_contribution", "Region contributions"),
        (patches_gdf, "combined_score", "Combined (patch × region)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (gdf, col, panel_title) in zip(axes, panels):
        scores = gdf[col].values.astype(np.float64)
        cm_obj = plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())

        if thumbnail is not None:
            ax.imshow(thumbnail, origin="upper")
        else:
            bounds = gdf.total_bounds
            native_mpp_eff = native_mpp or 0.5
            s = (native_mpp_eff / thumbnail_mpp)
            ax.set_xlim(0, max(1, int(bounds[2] * s)))
            ax.set_ylim(max(1, int(bounds[3] * s)), 0)

        for geom, score in zip(gdf.geometry, scores):
            x0, y0, x1, y1 = geom.bounds
            rect = mpatches.Rectangle(
                (x0 * scale, y0 * scale),
                (x1 - x0) * scale,
                (y1 - y0) * scale,
                linewidth=0,
                facecolor=cm_obj(norm(score)),
                alpha=alpha,
            )
            ax.add_patch(rect)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title(panel_title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    if slide_id:
        fig.suptitle(slide_id, fontsize=13, y=1.01)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_cell_tokens(
    h5_path: Union[str, Path],
    attn_dict: Dict,
    wsi_path: Optional[Union[str, Path]] = None,
    top_k: int = 16,
    grid_cols: int = 4,
    cmap: str = "hot",
    alpha: float = 0.55,
    patch_size_px: int = 128,
    figsize_per_patch: float = 2.5,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Visualize cell-token attention heatmaps for the top-k attended patches.

    For each of the top-k patches (by PatchExpert attention), overlays a
    resized ``(sqrt_T × sqrt_T)`` token attention heatmap on the actual
    patch image (read from the WSI if *wsi_path* is provided, otherwise
    the heatmap alone is shown on a grey canvas).

    Parameters
    ----------
    h5_path : str or Path
        Hierarchical H5 file (needed to read ``mid/xy_256`` bboxes).
    attn_dict : dict
        Normalised attention dict.  Must contain ``'cell'`` (hierarchical).
    wsi_path : str or Path, optional
        Raw WSI file.  Required to render the actual H&E patch images.
    top_k : int
        Number of highest-attention patches to show.
    grid_cols : int
        Columns in the output grid.
    cmap : str
        Colormap for the token attention heatmap overlay.
    alpha : float
        Opacity of the heatmap overlay on the patch image.
    patch_size_px : int
        Display size (pixels) for each patch thumbnail in the grid.
    figsize_per_patch : float
        Inches allocated per patch in the figure.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Requires ``'cell'`` in *attn_dict* (hierarchical model).  For additive
    models this function is not applicable and raises ``KeyError``.
    """
    from math import isqrt

    if "cell" not in attn_dict:
        raise KeyError(
            "'cell' key not found in attn_dict.  "
            "plot_cell_tokens requires a hierarchical model output."
        )

    meta = read_h5_spatial_metadata(h5_path)
    xy_256 = meta["xy_256"]            # (N, 4)
    mpp = meta["mpp"]

    # Re-derive top-k patch indices
    top_k_idx = get_top_k_patch_indices(attn_dict, k=top_k)
    top_k_idx_np = top_k_idx.numpy()

    # Cell expert attention: (K*T, C) — sorted by PatchExpert top-k order
    cell_attn = attn_dict["cell"]["attn_weights"]  # (K*T, C)
    if isinstance(cell_attn, torch.Tensor):
        cell_attn = cell_attn.numpy()

    K = len(top_k_idx_np)
    total_cell_tokens = cell_attn.shape[0]
    T = total_cell_tokens // K if K > 0 else 256
    grid_size = isqrt(T)  # e.g. 4 for T=16, 16 for T=256

    n_rows = (K + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(
        n_rows,
        grid_cols,
        figsize=(grid_cols * figsize_per_patch, n_rows * figsize_per_patch),
    )
    axes = np.array(axes).reshape(-1)

    for i, patch_idx in enumerate(top_k_idx_np):
        ax = axes[i]

        # Extract token attention for this patch and reshape to grid
        token_attn = cell_attn[i * T : (i + 1) * T].mean(axis=-1)  # (T,)
        heatmap = token_attn.reshape(grid_size, grid_size)

        # Resize heatmap to patch_size_px × patch_size_px
        from PIL import Image as PILImage
        heatmap_img = PILImage.fromarray(
            (normalize_scores(heatmap) * 255).astype(np.uint8)
        ).resize((patch_size_px, patch_size_px), PILImage.NEAREST)
        heatmap_np = np.asarray(heatmap_img)

        # Try to read the actual patch image from WSI
        patch_rgb = None
        if wsi_path is not None:
            try:
                patch_rgb = _read_patch_from_wsi(wsi_path, xy_256[patch_idx], patch_size_px)
            except Exception:
                pass

        if patch_rgb is not None:
            ax.imshow(patch_rgb)
            # Overlay heatmap
            cm_cell = plt.get_cmap(cmap)
            heatmap_rgba = cm_cell(normalize_scores(heatmap_np.astype(np.float64) / 255.0))
            heatmap_rgba[..., 3] = alpha
            ax.imshow(heatmap_rgba, origin="upper")
        else:
            # No WSI: show heatmap alone
            ax.imshow(heatmap_np, cmap=cmap, vmin=0, vmax=255)

        ax.set_title(f"Patch {patch_idx}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for j in range(K, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Cell-token attention — top-{K} patches", fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def _read_patch_from_wsi(
    wsi_path: Union[str, Path],
    bbox: np.ndarray,
    output_px: int = 128,
) -> Optional[np.ndarray]:
    """Read a single patch from a WSI and resize to ``output_px × output_px``.

    Bounding box coordinates are expected in **native level-0 pixel space**
    (as stored in ``xy_256``).  For scanners whose native MPP is < 0.35
    (typically 0.25 MPP / 40×) the image is read from pyramid level 1 with
    halved dimensions, mirroring the tiling logic in
    ``extract_tiles_from_wsi``, so the returned crop always corresponds to
    the ~0.5 MPP tiling resolution.

    Parameters
    ----------
    wsi_path : str or Path
        Raw WSI file path.
    bbox : np.ndarray, shape ``(4,)``
        ``[xmin, ymin, xmax, ymax]`` in native level-0 pixel coordinates.
    output_px : int
        Size of the returned patch in pixels.

    Returns
    -------
    np.ndarray ``(output_px, output_px, 3)`` uint8, or ``None`` on failure.
    """
    from PIL import Image as PILImage

    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    w = xmax - xmin
    h = ymax - ymin

    try:
        import openslide
        slide = openslide.open_slide(str(wsi_path))
        scanner_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5))
        if scanner_mpp < 0.35:   # ~0.25 MPP scanner — read at 0.5 MPP level
            target_level, rw, rh = 1, w // 2, h // 2
        else:
            target_level, rw, rh = 0, w, h
        region = slide.read_region((xmin, ymin), target_level, (rw, rh)).convert("RGB")
        region = region.resize((output_px, output_px), PILImage.LANCZOS)
        return np.asarray(region)
    except Exception:
        pass

    try:
        from wsidata import open_wsi
        wsi = open_wsi(str(wsi_path))
        scanner_mpp = float(wsi.properties.get("mpp", 0.5))
        if scanner_mpp < 0.35:   # ~0.25 MPP scanner — read at 0.5 MPP level
            target_level, rw, rh = 1, w // 2, h // 2
        else:
            target_level, rw, rh = 0, w, h
        region = wsi.read_region((xmin, ymin), size=(rw, rh), level=target_level)
        region_img = PILImage.fromarray(region).resize(
            (output_px, output_px), PILImage.LANCZOS
        )
        return np.asarray(region_img)
    except Exception:
        return None


# ======================================================================
# 7.  Interactive napari viewer
# ======================================================================

def launch_napari_viewer(
    patches_gdf: gpd.GeoDataFrame,
    wsi_path: Optional[Union[str, Path]] = None,
    regions_gdf: Optional[gpd.GeoDataFrame] = None,
    score_cols: Optional[List[str]] = None,
    native_mpp: Optional[float] = None,
    thumbnail_mpp: float = 4.0,
    cmap: str = "RdBu_r",
    viewer_title: str = "MIL Attention Viewer",
):
    """Launch an interactive napari viewer with prediction overlays.

    Adds the following layers:
        * **Image**   — slide thumbnail (requires *wsi_path*).
        * **Shapes**  — patch bboxes, face color mapped to each score
          column in *score_cols*.
        * **Shapes**  — region bboxes (optional; hierarchical only).

    Parameters
    ----------
    patches_gdf : GeoDataFrame
        Patch-level GeoDataFrame (output of :func:`build_patch_geodataframe`).
    wsi_path : str or Path, optional
        Raw WSI file.  Required for the background image layer.
    regions_gdf : GeoDataFrame, optional
        Region-level GeoDataFrame (hierarchical models).
    score_cols : list of str, optional
        Columns to visualize as separate color layers.  Defaults to all
        numeric columns except ``'geometry'``.
    native_mpp : float, optional
        Native tiling mpp.  Used to compute scale to thumbnail coordinates.
    thumbnail_mpp : float
        MPP at which the background thumbnail is rendered.
    cmap : str
        Colormap name (napari/matplotlib compatible).
    viewer_title : str
        Title displayed in the napari window.

    Returns
    -------
    napari.Viewer
        The opened viewer instance (keep a reference to prevent GC).

    Raises
    ------
    RuntimeError
        If no display is available (headless environment).
    ImportError
        If *napari* is not installed.
    """
    # Guard: headless environment check
    _check_display_available()

    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is not installed.  Install with: pip install napari[all]"
        ) from exc

    viewer = napari.Viewer(title=viewer_title)

    # Scale factor: native pixels → thumbnail pixels
    if native_mpp is None:
        native_mpp = 0.5
    scale = native_mpp / thumbnail_mpp

    # Background image
    if wsi_path is not None:
        thumbnail, scale = get_slide_thumbnail(
            wsi_path, mpp=thumbnail_mpp, native_mpp=native_mpp
        )
        viewer.add_image(thumbnail, name="H&E", rgb=True)

    # Determine score columns
    if score_cols is None:
        score_cols = [
            c for c in patches_gdf.columns
            if c != "geometry" and np.issubdtype(patches_gdf[c].dtype, np.floating)
        ]

    for col in score_cols:
        scores = patches_gdf[col].values.astype(np.float32)
        rects = []
        for geom in patches_gdf.geometry:
            x0, y0, x1, y1 = geom.bounds
            # napari shapes: [[y_start, x_start], [y_end, x_end]] (row, col)
            rects.append(
                np.array([
                    [y0 * scale, x0 * scale],
                    [y0 * scale, x1 * scale],
                    [y1 * scale, x1 * scale],
                    [y1 * scale, x0 * scale],
                ])
            )
        viewer.add_shapes(
            rects,
            shape_type="rectangle",
            face_color=_scores_to_rgba(scores, cmap),
            edge_width=0,
            name=f"patches — {col}",
            opacity=0.6,
        )

    # Region overlay
    if regions_gdf is not None:
        reg_scores = regions_gdf["region_contribution"].values.astype(np.float32)
        reg_rects = []
        for geom in regions_gdf.geometry:
            x0, y0, x1, y1 = geom.bounds
            reg_rects.append(
                np.array([
                    [y0 * scale, x0 * scale],
                    [y0 * scale, x1 * scale],
                    [y1 * scale, x1 * scale],
                    [y1 * scale, x0 * scale],
                ])
            )
        viewer.add_shapes(
            reg_rects,
            shape_type="rectangle",
            face_color=_scores_to_rgba(reg_scores, cmap),
            edge_width=1,
            edge_color="white",
            name="regions — region_contribution",
            opacity=0.35,
        )

    return viewer


def _scores_to_rgba(scores: np.ndarray, cmap: str) -> np.ndarray:
    """Map 1-D scores to RGBA colors using *cmap*.

    Returns
    -------
    np.ndarray ``(N, 4)`` float32
    """
    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    cm_obj = plt.get_cmap(cmap)
    return cm_obj(norm(scores)).astype(np.float32)


def _check_display_available() -> None:
    """Raise ``RuntimeError`` if no display is available (headless node)."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    # JupyterHub / VSCode with forwarding may not set DISPLAY but work fine
    # Only raise if we're certain we're headless (no DISPLAY, no notebook)
    try:
        import IPython
        shell = IPython.get_ipython()
        if shell is not None:
            return  # We're in a Jupyter kernel — assume display works
    except ImportError:
        pass
    raise RuntimeError(
        "No display detected (DISPLAY / WAYLAND_DISPLAY not set).  "
        "Run napari from a JupyterHub session or forward your X display.  "
        "Use the static matplotlib functions (plot_patch_scores etc.) for "
        "headless visualization."
    )
