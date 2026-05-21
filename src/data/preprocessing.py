"""
Functions for data preprocessing
"""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import os
from typing import Any, Dict, Callable, Sequence
import torch
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide._const import Key
from lazyslide._utils import default_pbar
from lazyslide.models import MODEL_REGISTRY, ImageModel, list_models

import numpy as np
import geopandas as gpd
from shapely import box
from spatialdata.models import ShapesModel
from wsidata import TileSpec
from shapely.ops import unary_union
import geopandas as gpd
from PIL import Image
import h5py

class BackgroundFilterFailure(Exception):
    """
    Raised when background filtering produces no valid tissue tiles
    indicating the slide may be too faded for RGB-based filtering.
    Can happen especially for Newcastle
    """
    pass

def create_subtiles_from_region_tiles(
        wsi, 
        region_tile_key="region_tiles",
        subtile_px=256,
        mpp=0.5,
        background_fraction=0.5,
        key_added="tiles",
        return_tiles=False,
        edge=True
):
    """
    Create sub-tiles from region tiles that overlap with tissue regions.

    For each region-level tile (e.g. 2048×2048), generates a grid of smaller
    sub-tiles (e.g. 256×256) at the same MPP. Each sub-tile records which
    region tile it came from via a ``region_tile_id`` column.

    Parameters
    ----------
    wsi : WSIData
        The WSIData object containing the region tiles.
    region_tile_key : str, default "region_tiles"
        Key in ``wsi.shapes`` where the region tiles are stored.
    subtile_px : int, default 256
        Size of each sub-tile in pixels (at the requested mpp).
    mpp : float, default 0.5
        Microns per pixel for the sub-tiles.
    background_fraction : float, default 0.5
        Maximum allowed background fraction; sub-tiles with more background
        are discarded.  Set to 1.0 to skip filtering.
    edge : bool, default True
        Whether to include edge sub-tiles that extend beyond the region
        boundary (only relevant when the region size is not an exact
        multiple of subtile_px).
    key_added : str, default "tiles"
        Key under which the sub-tiles are stored in ``wsi.shapes``.
    return_tiles : bool, default False
        If True, return the sub-tiles GeoDataFrame and TileSpec.
    edge: bool, default True
        Whether to include edge tiles that may extend beyond the region boundary.
    Returns
    -------
    (GeoDataFrame, TileSpec) or None
    """
    # build tile spec
    subtile_spec = TileSpec.from_wsidata(
        wsi,
        tile_px=subtile_px,
        mpp=mpp,
        tissue_name=None
    )
    
    # Get base width and height
    subtile_base_w = subtile_spec.base_width
    subtile_base_h = subtile_spec.base_height

    region_tiles = wsi.shapes[region_tile_key]

    all_subtiles = []
    all_region_ids = []

    for _, region_row in region_tiles.iterrows():
        region_id = region_row['tile_id']
        minx, miny, maxx, maxy = region_row.geometry.bounds

        region_w = maxx - minx
        region_h = maxy - miny

        n_cols = int(region_w // subtile_base_w)
        n_rows = int(region_h // subtile_base_h)

        # Include partial tiles if requested
        if edge and region_h % subtile_base_h !=0:
            n_rows += 1
        if edge and region_w % subtile_base_w != 0:
            n_cols += 1
        
        for r in range(n_rows):
            for c in range(n_cols):
                sx = minx + c * subtile_base_w
                sy = miny + r * subtile_base_h
                all_subtiles.append(box(sx, sy, sx + subtile_base_w, sy + subtile_base_h))
                all_region_ids.append(region_id)

    subtiles_gdf = gpd.GeoDataFrame({
        'geometry': all_subtiles,
        f'{region_tile_key}_id': all_region_ids
    })

    original_len = len(subtiles_gdf)

    # filter for background
    if background_fraction < 1.0:
        subtiles_gdf = _filter_background_tiles(
            wsi, subtiles_gdf, subtile_px,  background_fraction
        )
    
    # Assign sequential IDs
    subtiles_gdf = subtiles_gdf.reset_index(drop=True)
    subtiles_gdf['tile_id'] = np.arange(len(subtiles_gdf))
    subtiles_gdf = subtiles_gdf[['tile_id', f'{region_tile_key}_id', 'geometry']]
    
    # Store in the WSI object
    if key_added is not None:
        wsi.shapes[key_added] = ShapesModel.parse(subtiles_gdf)
        
        if wsi.TILE_SPEC_KEY in wsi.attrs:
            spec_data = wsi.attrs[wsi.TILE_SPEC_KEY]
            spec_data[key_added] = subtile_spec.to_dict()
        else:
            spec_data = {key_added: subtile_spec.to_dict()}
            wsi.attrs[wsi.TILE_SPEC_KEY] = spec_data
    
    print(
        f"Created {len(subtiles_gdf)} sub-tiles ({subtile_px}px), filtered from {original_len} sub-tiles"
        f"from {len(region_tiles)} region tiles"
    )
    
    if return_tiles:
        return subtiles_gdf, subtile_spec
    return None

def tile_whole_wsi_with_background_filter(
    wsi,
    tile_px: int | tuple[int, int],
    *,
    stride_px: int | tuple[int, int] = None,
    overlap: float = None,
    edge: bool = False,
    mpp: float = None,
    slide_mpp: float = None,
    ops_level: int = None,
    background_fraction: float = 0.3,
    key_added: str = "tiles",
    return_tiles: bool = False,
):
    """
    Generate tiles for the whole WSI with background filtering.
    
    Parameters
    ----------
    wsi : WSIData
        The WSIData object to work on.
    tile_px : int or (int, int)
        The size of the tile, if tuple, (W, H).
    stride_px : int or (int, int), optional
        The stride of the tile, if tuple, (W, H).
    overlap : float, optional
        The overlap of the tiles, exclusive with stride_px.
    edge : bool, default False
        Whether to include edge tiles.
    mpp : float, optional
        The requested mpp of the tiles.
    slide_mpp : float, optional
        Override the slide mpp.
    ops_level : int, optional
        Which level to use for tile retrieval.
    background_fraction : float, default 0.3
        Maximum fraction of background allowed in a tile.
        Tiles with more background than this will be filtered out.
    key_added : str, default 'tiles'
        The key to store tiles in wsi.shapes.
    return_tiles : bool, default False
        Return the tiles dataframe.
    
    Returns
    -------
    tiles : GeoDataFrame (if return_tiles=True)
        The tiles dataframe with columns tile_id and geometry.
    tile_spec : TileSpec (if return_tiles=True)
        The tile specification.
    """
    
    # Create the tile spec
    tile_spec = TileSpec.from_wsidata(
        wsi,
        tile_px=tile_px,
        stride_px=stride_px,
        overlap=overlap,
        mpp=mpp,
        ops_level=ops_level,
        slide_mpp=slide_mpp,
        tissue_name=None,
    )
    
    # Get the whole slide bounds
    x, y, w, h = wsi.properties.bounds
    
    # Generate all tiles
    tiles = tiles_from_bbox(
        x,
        y,
        w,
        h,
        tile_spec.base_width,
        tile_spec.base_height,
        stride_w=tile_spec.base_stride_width,
        stride_h=tile_spec.base_stride_height,
        edge=edge,
    )
    
    # Apply background filtering
    if background_fraction < 1.0:
        tiles = _filter_background_tiles(wsi, tiles, tile_px, background_fraction)
    
    # Add tile IDs
    tiles["tile_id"] = np.arange(len(tiles))
    tiles = tiles[["tile_id", "geometry"]]
    
    # Add to WSI object
    if key_added is not None:
        wsi.shapes[key_added] = ShapesModel.parse(tiles)
        
        if wsi.TILE_SPEC_KEY in wsi.attrs:
            spec_data = wsi.attrs[wsi.TILE_SPEC_KEY]
            spec_data[key_added] = tile_spec.to_dict()
        else:
            spec_data = {key_added: tile_spec.to_dict()}
            wsi.attrs[wsi.TILE_SPEC_KEY] = spec_data
    
    if return_tiles:
        return tiles, tile_spec
    return None

def tiles_from_bbox(
    x,
    y,
    w,
    h,
    tile_w: int,
    tile_h: int,
    stride_w=None,
    stride_h=None,
    edge=True,
    mask=None,
):
    """Create tiles from a bounding box.

    Parameters
    ----------
    x, y, w, h : int
        The x, y, width, height of the bounding box.
    tile_w, tile_h: int
        The width/height of tiles.
    stride_w, stride_h : int, default None
        The width/height of stride when moving to the next tile.
    edge : bool, default True
        Whether to include the edge tiles.
    mask : Polygon, default None
        The mask to use for the tiles.

    Returns
    -------
    List[Polygon]
        The list of tiles.

    """
    # A new implementation in pure numpy and return shapely geometry
    x, y, w, h = int(x), int(y), int(w), int(h)

    if stride_w is None:
        stride_w = tile_w
    if stride_h is None:
        stride_h = tile_h

    # calculate the number of expected tiles
    # If the width/height is divisible by stride,
    # We need to add 1 to include the starting point
    nw = w // stride_w + 1
    nh = h // stride_h + 1

    # To include the edge tiles
    if edge and w % stride_w != 0:
        nw += 1
    if edge and h % stride_h != 0:
        nh += 1

    xs = np.arange(nw, dtype=np.uint) * stride_w + x
    ys = np.arange(nh, dtype=np.uint) * stride_h + y
    # points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    # Add xs and ys after stride
    if stride_h != tile_h:
        yss = ys + tile_h
        yss = np.sort(np.unique(np.concatenate((ys, yss))))
    else:
        yss = ys
    if stride_w != tile_w:
        xss = xs + tile_w
        xss = np.sort(np.unique(np.concatenate((xs, xss))))
    else:
        xss = xs

    tiles = []
    pt_counts = []
    if mask is not None:
        # Filter the points that are within the mask
        tile_points = np.array(np.meshgrid(xss, yss)).T.reshape(-1, 2)
        prepare(mask)
        is_in = contains_xy(mask, x=tile_points[:, 0], y=tile_points[:, 1])
        # make a dict mapping if the point is in the mask
        in_dict = dict(zip(map(tuple, tile_points), is_in))
        for i in range(nw):
            for j in range(nh):
                x, y = xs[i], ys[j]
                p1, p2, p3, p4 = (
                    (x, y),
                    (x + tile_w, y),
                    (x + tile_w, y + tile_h),
                    (x, y + tile_h),
                )
                pt_count = sum(in_dict.get(p, 0) for p in (p1, p2, p3, p4))
                if pt_count > 0:
                    tiles.append(box(x, y, x + tile_w, y + tile_h))
                    pt_counts.append(pt_count)
    else:
        for i in range(nw):
            for j in range(nh):
                x, y = xs[i], ys[j]
                tiles.append(box(x, y, x + tile_w, y + tile_h))
        pt_counts = 4
    return gpd.GeoDataFrame({"geometry": tiles, "pt_count": pt_counts})


def _filter_background_tiles(wsi, tiles, tile_px, background_fraction):
    """
    Filter tiles based on background content.
    
    This uses a low-resolution thumbnail to estimate background content.
    """
    # Get a low-res thumbnail for background detection
    # Adjust the level or size based on your needs
    thumb = wsi.get_thumbnail(size=tile_px)  # Adjust size as needed
    thumb_array = np.array(thumb)

    # Simple background detection
    # For H&E slides, background is often white/light
    # This is a simple threshold - adjust based on your staining
    if thumb_array.ndim == 2:
        # Grayscale thumbnail — no axis=2 available
        print(
            "[_filter_background_tiles] WARNING: thumbnail is grayscale (ndim=2), "
            "using it directly for background detection",
            flush=True,
        )
        is_background = thumb_array > 190
    else:
        is_background = np.mean(thumb_array, axis=2) > 190  # White background
    
    # Calculate the scale factor between thumbnail and base resolution
    thumb_h, thumb_w = is_background.shape
    base_h, base_w = wsi.properties.shape[0], wsi.properties.shape[1]
    scale_y = thumb_h / base_h
    scale_x = thumb_w / base_w
    
    # Filter tiles
    filtered_tiles = []
    all_bg_fracs = []
    n_zero_area = 0
    for idx, row in tiles.iterrows():
        geom = row["geometry"]
        minx, miny, maxx, maxy = geom.bounds
        
        # Convert to thumbnail coordinates
        tx1 = int(minx * scale_x)
        ty1 = int(miny * scale_y)
        tx2 = int(maxx * scale_x)
        ty2 = int(maxy * scale_y)
        
        # Ensure within bounds
        tx1 = max(0, min(tx1, thumb_w - 1))
        tx2 = max(0, min(tx2, thumb_w))
        ty1 = max(0, min(ty1, thumb_h - 1))
        ty2 = max(0, min(ty2, thumb_h))
        
        # Calculate background fraction in this tile
        if tx2 > tx1 and ty2 > ty1:
            tile_region = is_background[ty1:ty2, tx1:tx2]
            bg_frac = np.mean(tile_region)
            all_bg_fracs.append(bg_frac)
            
            # Keep tile if background fraction is below threshold
            if bg_frac <= background_fraction:
                filtered_tiles.append(row)
        else:
            n_zero_area += 1

    # --- Defensive fallback (Phase 3) ------------------------------------
    # If every tile was discarded (e.g. wrong scale factors, unusual thumbnail),
    # skip the filter entirely and let the downstream tissue-mask intersection
    # handle background removal instead.
    if len(filtered_tiles) == 0:
        raise BackgroundFilterFailure(
            "All tiles are background based on the filtering" \
            "Probably the image is too faded"
        )
    return gpd.GeoDataFrame(filtered_tiles, crs=tiles.crs)


def filter_tiles_by_tissue_mask(wsi, tile_key="tiles", tissue_key="tissues"):
    """
    Filter tiles to keep only those that overlap with tissue polygons.
    
    This assumes that the tissue polygons are stored in wsi['tissues'] and the tiles are in wsi['tiles'].
    It updates wsi['tiles'] to keep only those that intersect with any tissue polygon.
    """
    # Filter tiles to keep only those that overlap with tissue polygons
    if len(wsi[tissue_key]) > 0:
        original_tiles= len(wsi[tile_key])
        # Create union of all tissue polygons for efficient intersection checking
        # tissue_union = unary_union(wsi['tissues'].geometry)

        # Perform spatial join to find tiles that intersect with tissues
        filtered_tiles = wsi[tile_key].sjoin(
            wsi[tissue_key], 
            how='inner',
            predicate='intersects'
        )

        # Remove duplicate tiles (in case a tile intersects multiple tissue polygons)
        # Keep all original columns including tile_id, pt_count, tissue_id
        filtered_tiles = filtered_tiles.drop(columns=['index_right']).drop_duplicates(subset=['tile_id'])

        if len(filtered_tiles) == 0:
            raise BackgroundFilterFailure(
                f"No tiles overlap any tissue region after spatial join — "
                f"background filtering likely discarded tissue tiles on a faded slide. "
                f"Re-run without background filtering."
            )

        # Update wsi with filtered tiles
        wsi[tile_key] = filtered_tiles
    else:
        # No tissues found, keep empty tiles or handle as needed
        wsi[tile_key] = wsi[tile_key].iloc[0:0]  # Empty GeoDataFrame with same structure

    print(f"Filtered tiles: {len(wsi[tile_key])} tiles overlap with tissue regions, original tiles were {original_tiles}")

def extract_tiles_from_wsi(wsi, output_dir, target_mpp):
    """
    Extract tiles from WSI based on polygon geometries at a specific MPP.
    
    Parameters:
    -----------
    wsi : WSIData
        WSI object from wsidata package
    output_dir : str or Path
        Directory to save extracted tiles
    target_mpp : float
        Target microns per pixel for extraction (must match segmentation MPP)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = wsi['tiles']

    # get mpp and levels to decide if donwscale
    mpp = wsi.properties.mpp
    downsampling_factor = wsi.fetch.pyramids()['downsample']
    
    for idx, row in gdf.iterrows():
        polygon = row.geometry
        # Get bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        
        # Convert to integer coordinates
        x = int(minx)
        y = int(miny)
        width = int(maxx - minx)
        height = int(maxy - miny)

        # We need to be careful when extracting imgs as the tiles represent the size at the specified mpp but the coords are always at level 0
        # This creates a problem if level 0 is lower than the target mpp, as the coordinates will be off. 
        # For example, if we harmonized at 0.5 but level 0 is 0.25, then the coordinates will be at 0.25 and we need to downsample by a factor of 2 to get the correct region at 0.5 mpp.
        # The reading function though accepts coordinates at level 0 and a level to read from. 
        # The coordinates and level extract correctly, but the width and height need to be adjusted accordingly as they need to 
        # show 0.5 mpp. If the original resolution is 0.5mpp, there is no problem at all.

        if mpp < 0.35:
            # The resolution is lower than 0.5, then we need to downsample as level 0 i ~ 0.25
            target_level = 1 # downsample to level 1 == 0.5 mpp, this assumes that level 0 is ~ 0.25 and level 1 is ~0.50
            # Resize width and height
            width = int(width//2)
            height = int(height//2)

        else:
            target_level = 0 # lowest possible level, usually 0.5 mpp, where donwsampling is 1.0
            # No need to resize width and height
        
        # Read the region from WSI at the specified level
        tile_image = wsi.read_region(
            x=x, # x and y needs to be supplied at the level 0 everytime.
            y=y,
            width=width,
            height=height,
            level=target_level
        )
        
        # Convert to PIL Image if needed
        if isinstance(tile_image, np.ndarray):
            tile_image = Image.fromarray(tile_image)
        
        # If the image has an alpha channel, convert to RGB
        if tile_image.mode == 'RGBA':
            tile_image = tile_image.convert('RGB')
        
        # Save as PNG
        output_path = output_dir / f"tile_{x}x_{y}y.png"
        tile_image.save(output_path, 'PNG')


def reindex_tiles_and_regions(
    wsi,
    region_tile_key: str = "region_tiles",
    subtile_key: str = "tiles",
) -> None:
    """Re-index region tiles and sub-tiles to contiguous IDs.

    After :func:`filter_tiles_by_tissue_mask`, region ``tile_id`` values
    may contain gaps (e.g. ``[0, 3, 7, 12, …]``).  This function remaps
    them to ``0 … N-1`` so that ``region_tile_id`` in the sub-tiles is a
    valid positional index into the region array (important for the
    output H5 layout).

    The sub-tile ``tile_id`` is also reset to ``0 … M-1`` and
    ``region_tile_id`` is remapped accordingly.

    Both GeoDataFrames are updated **in-place** on the ``wsi`` object.

    Parameters
    ----------
    wsi : WSIData
        The WSIData object containing both region and sub-tile shapes.
    region_tile_key : str, default "region_tiles"
        Key in ``wsi.shapes`` where the region tiles are stored.
    subtile_key : str, default "tiles"
        Key in ``wsi.shapes`` where the sub-tiles are stored.
    """
    # --- Region tiles ---------------------------------------------------
    region_tiles = wsi[region_tile_key].copy()
    old_region_ids = region_tiles['tile_id'].values
    new_region_ids = np.arange(len(region_tiles))
    old_to_new = dict(zip(old_region_ids, new_region_ids))

    region_tiles = region_tiles.reset_index(drop=True)
    region_tiles['tile_id'] = new_region_ids
    wsi[region_tile_key] = region_tiles

    # --- Sub-tiles ------------------------------------------------------
    subtiles = wsi[subtile_key].copy()
    subtiles[f'{region_tile_key}_id'] = (
        subtiles[f'{region_tile_key}_id'].map(old_to_new).astype(np.int32)
    )
    subtiles = subtiles.reset_index(drop=True)
    subtiles['tile_id'] = np.arange(len(subtiles))
    wsi[subtile_key] = subtiles

    # print(
    #     f'Re-indexed {len(region_tiles)} region tiles and '
    #     f'{len(subtiles)} sub-tiles to contiguous IDs',
    #     flush=True,
    # )

def save_tile_region_h5(
        wsi: Any, 
        outdir_features: str,
        model_name: str,
        tile_size: int, 
        slide_name: str,
        region_key: str = "region_tiles",
        subtile_key: str = "tiles",
        embd_dim: int = 2560,
        pool_cell_tokens: bool = True,  # if True, /cells/H shape is (N, 64, D); else (N, 256, D)
) -> str:
    """
    Save tile and region metadata to an H5 file *before* feature extraction.

    Layout
    ------
    /mid/tile_id    : (n_patches,)        int32
    /mid/region_id  : (n_patches,)        int32   patch→region map
    /mid/xy_256     : (n_patches, 4)      int32   [minx, miny, maxx, maxy]
    /mid/H_patch    : (n_patches, 1280)   float16  CLS token — PatchExpert input
    /mid/H_region   : (n_patches, 2560)   float16  CLS+mean — RegionExpert input
    /cells/H        : (n_patches, 256, 1280)  float16  dense tokens — CellExpert input
    /regions/region_id : (n_regions,)     int32
    /regions/xy_2048   : (n_regions, 4)   int32   [minx, miny, maxx, maxy]
    /slide_metadata/mpp : (1,)   float16

    """

    out_subdir = os.path.join(outdir_features, f'{model_name}_{tile_size}_hierarchical')
    os.makedirs(out_subdir, exist_ok=True)
    h5_path = os.path.join(out_subdir, f"{slide_name}.h5")

    wsi.attrs['H5_path'] = h5_path

    # get mpp of the slide for later use in fetch_fn
    mpp = wsi.properties.mpp

    # ── Region metadata ──────────────────────────────────────────────
    region_gdf = wsi[region_key]
    region_ids = region_gdf['tile_id'].values.astype(np.int32)
    region_bounds = np.array(
        [geom.bounds for geom in region_gdf.geometry], dtype=np.int32
    )  # (n_regions, 4) → [minx, miny, maxx, maxy]

    # ── Sub-tile metadata ────────────────────────────────────────────
    tiles_gdf = wsi[subtile_key]
    tile_ids = tiles_gdf['tile_id'].values.astype(np.int32)
    tile_region_ids = tiles_gdf[f'{region_key}_id'].values.astype(np.int32)
    tile_bounds = np.array(
        [geom.bounds for geom in tiles_gdf.geometry], dtype=np.int32
    )  # (n_patches, 4)

    n_patches = len(tiles_gdf)
    emb_dim = embd_dim  # FM embedding dimension

    # ── Write H5 ─────────────────────────────────────────────────────
    with h5py.File(h5_path, 'w') as h5f:
        # /regions — metadata only; region aggregation is done inside RegionExpert
        grp_reg = h5f.create_group('regions')
        grp_reg.create_dataset('region_id', data=region_ids)   # (R,)
        grp_reg.create_dataset('xy_2048', data=region_bounds)  # (R, 4)

        # /mid
        grp_mid = h5f.create_group('mid')
        grp_mid.create_dataset('tile_id', data=tile_ids)            # (N,)
        grp_mid.create_dataset('region_id', data=tile_region_ids)   # (N,) patch→region map
        grp_mid.create_dataset('xy_256', data=tile_bounds)          # (N, 4)
        # CLS token only — PatchExpert input
        grp_mid.create_dataset(
            'H_patch',
            shape=(n_patches, emb_dim // 2),  # (N, 1280)
            dtype=np.float16,
            fillvalue=0.0,
        )
        # CLS + mean(patch tokens) — RegionExpert input, aggregated by region_id in model
        grp_mid.create_dataset(
            'H_region',
            shape=(n_patches, emb_dim),  # (N, 2560)
            dtype=np.float16,
            fillvalue=0.0,
        )

        # /cells — dense token grids, CellExpert input
        # pool_cell_tokens=True: 4×4 avg pool during extraction → (N, 16, 1280) @ 32 μm/token
        # pool_cell_tokens=False: raw backbone output            → (N, 256, 1280) @ 8 μm/token
        n_cell_tokens = 16 if pool_cell_tokens else 256
        grp_cells = h5f.create_group('cells')
        grp_cells.create_dataset(
            'H',
            shape=(n_patches, n_cell_tokens, emb_dim // 2),
            dtype=np.float16,
            fillvalue=0.0,
            chunks=(1, n_cell_tokens, emb_dim // 2),  # row-wise chunking for fast random access
        )

        # /slide_metadata
        grp_slide = h5f.create_group('slide_metadata')
        grp_slide.create_dataset('mpp', data=np.array([mpp], dtype=np.float16))

    print(
        f'Saved tile/region metadata to {h5_path}  '
        f'(regions={len(region_ids)}, patches={n_patches})',
        flush=True,
    )

def save_embeddings_to_h5(
        wsi: Any,
        model_name: str,
        region: np.ndarray,
        patch: np.ndarray,
        cell: np.ndarray
) -> None:
    """
    Write feature embeddings into the pre-allocated datasets:
    - /mid/H_region  : per-patch CLS+mean embeddings (N, 2560) for RegionExpert
    - /mid/H_patch   : per-patch CLS embeddings (N, 1280) for PatchExpert
    - /cells/H       : per-patch dense token grids (N, 256, 1280) for CellExpert

    """
    # H5 path is store in wsi.attrs
    h5_path = wsi.attrs['H5_path']

    region_embeddings = np.array(region, dtype=np.float16)  # (n_patches, 2560)
    patch_embeddings = np.array(patch, dtype=np.float16)  # (n_patches, 1280)
    cell_embeddings = np.array(cell, dtype=np.float16)  # (n_patches, 256, 1280)

    with h5py.File(h5_path, 'r+') as h5f:
        saved_n = h5f['mid/tile_id'].shape[0]
        assert patch_embeddings.shape[0] == saved_n, (
            f"Embedding row count ({patch_embeddings.shape[0]}) != "
            f"saved tile count ({saved_n}). Possible reordering."
        )
        assert region_embeddings.shape[0] == saved_n, (
            f"Region embedding row count ({region_embeddings.shape[0]}) != "
            f"saved tile count ({saved_n}). Possible reordering."
        )
        h5f['mid/H_region'][:] = region_embeddings
        h5f['mid/H_patch'][:] = patch_embeddings
        h5f['cells/H'][:] = cell_embeddings

    print(
        f'Saved all embeddings to {h5_path}',
        flush=True,
    )


# Small adjustments for the lazyslide API
def feature_extraction(
    wsi: WSIData,
    model: str | Callable | ImageModel = None,
    model_path: str | Path = None,
    model_name: str = None,
    jit: bool = False,
    token: str = None,
    load_kws: dict = None,
    transform: Callable = None,
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    tile_key: str = Key.tiles,
    key_added: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pbar: bool = None,
    return_features: bool = False,
    pool_cell_tokens: bool = True,  # apply 2×2 avg pool to reduce (N,256,D)→(N,64,D) on GPU
    **kwargs,
):
    """
    Extract :term:`features` from :term:`WSI` :term:`tiles <tile>` using a pre-trained :term:`vision models <vision model>`.

    To list all timm models:

    .. code-block:: python

        >>> import timm
        >>> timm.list_models(pretrained=True)

    To list all lazyslide built-in models:

    .. code-block:: python

        >>> import lazyslide as zs
        >>> zs.models.list_models()

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    model : str or model object
        The model used for image :term:`feature extraction`.
        A list of built-in :term:`foundation models <foundation model>` can be found in :ref:`models-section`.
        Other models can be loaded from :term:`Hugging Face`, but only models with feature extraction head implemented.
    model_path : str or Path
        The path to the model file. Either model or model_path must be provided.
        If you don't have internet access, you can download the model file and load it from the local path.
        You can also load custom models from local files.
    model_name : str, optional
        If you provide your own model, you can specify the model name for the key_added.
        Or you can override the model name by providing a new model name.
    jit : bool, default: False
        Whether the model is a JIT model. If True, use torch.jit.load to load the model.
    token : str, optional
        The token for downloading the model from Hugging Face Hub for foundation models.
    load_kws : dict, optional
        Options to pass to the model creation function.
    transform : callable, optional
        The :term:`transform function` for the input image.
        If not provided, a default ImageNet transform function will be used.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    key_added : str, optional
        The key to store the extracted features.
    batch_size : int, optional
        The batch size for inference.
    num_workers : int, optional
        - mode='batch', The number of workers for data loading.
        - mode='chunk', The number of workers for parallel inference.
    pbar : bool, default: True
        Whether to show progress bar.
    return_features : bool, default: False
        Whether to return the extracted features.

    Returns
    -------
    :class:`numpy.ndarray` or None
        If return_features is True, return the extracted features.

    .. note::
        The feature matrix will be added to :code:`{model_name}_{tile_key}`
        in :bdg-danger:`tables` slot of :term:`WSIData` object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> wsi.fetch.features_anndata("resnet50")

    """

    device = device
    amp = amp
    autocast_dtype = autocast_dtype
    pbar = pbar

    load_kws = {} if load_kws is None else load_kws

    if model is not None:
        if isinstance(model, Callable):
            model = model
        elif isinstance(model, str):
            model, default_model_name = load_models(
                model_name=model, model_path=model_path, token=token, **load_kws
            )
            if model_name is None:
                model_name = default_model_name
        elif isinstance(model, ImageModel):
            model = model
            model_name = model.name
        else:
            raise ValueError("Model must be a model name or a model object.")
    else:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        model_path = Path(model_path)
        if model_path.exists():
            load_kws.setdefault("weights_only", False)
            load_func = torch.load if not jit else torch.jit.load
            model = load_func(model_path, **load_kws)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    # Deal with key_added
    if key_added is None:
        if model_name is not None:
            key_added = model_name
        elif isinstance(model, ImageModel):
            key_added = model.name
        elif hasattr(model, "__class__"):
            key_added = model.__class__.__name__
        elif hasattr(model, "__name__"):
            key_added = model.__name__
        else:
            key_added = "features"
        key_added = Key.feature(key_added, tile_key)
    try:
        model.to(device)
    except:  # noqa: E722
        pass

    if transform is None:
        if isinstance(model, ImageModel):
            transform = model.get_transform()
    # Create dataloader
    # Auto chunk the wsi tile coordinates to the number of workers'
    n_tiles = len(wsi.shapes[tile_key])

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=n_tiles)

        dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        # Extract features
        region_features = []
        patch_features = []
        cell_features = []
        if isinstance(device, torch.device):
            device = device.type
        amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
        with amp_ctx, torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device)
                if isinstance(model, ImageModel):
                    region_emb, patch_emb, cell_emb = model.encode_image(image) #output is a tuple of (concat(CLS+mean), CLS, patch_tokens)
                    output = patch_emb # let it track it for the rest of the pipeline
                else:
                    output = model(image)
                if not isinstance(output, np.ndarray):
                    output = output.cpu().numpy()
                
                if pool_cell_tokens:
                    # 4×4 avg pool on GPU: (B, 256, D) → (B, 16, D)
                    B, T, D = cell_emb.shape
                    side = int(T ** 0.5)  # 16
                    cell_grid = cell_emb.reshape(B, side, side, D).permute(0, 3, 1, 2).float()
                    cell_grid = torch.nn.functional.avg_pool2d(cell_grid, kernel_size=4, stride=4)
                    cell_emb = cell_grid.permute(0, 2, 3, 1).reshape(B, cell_grid.shape[2] * cell_grid.shape[3], D)
                region_features.append(region_emb.cpu().numpy())
                patch_features.append(patch_emb.cpu().numpy())
                cell_features.append(cell_emb.cpu().numpy())
                
                progress_bar.update(task, advance=len(image))
                del batch  # Free up memory
        # The progress bar may not reach 100% if exit too early
        # Force update
        progress_bar.refresh()
        region_feat = np.vstack(region_features)
        patch_feat = np.vstack(patch_features)
        cell_feat = np.vstack(cell_features)

    add_features(wsi, key=key_added, tile_key=tile_key, features=patch_feat)
    if return_features:
        return region_feat, patch_feat, cell_feat
    return None
