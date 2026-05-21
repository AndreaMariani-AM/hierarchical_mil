import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from typing import Tuple, Dict, Optional, List
import warnings
import ast
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def extract_nuclei_centroids(annotations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract nuclei detections and their precomputed centroids from annotations GeoDataFrame.
    
    Parameters
    ----------
    annotations_gdf : gpd.GeoDataFrame
        Annotations GeoDataFrame with columns: objectType, measurements, tissue_id, geometry
        Expected structure:
        - objectType: 'annotation' or 'detection'
        - measurements: dict containing 'Centroid_x' and 'Centroid_y' keys
        - tissue_id: identifier for tissue fragment
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame containing only detections with columns:
        - tissue_id, id, objectType, measurements, geometry
        - centroid_x: precomputed x-coordinate of nuclei centroid
        - centroid_y: precomputed y-coordinate of nuclei centroid
        
    Raises
    ------
    ValueError
        If no 'detection' objects found, or if Centroid_x/Centroid_y not in measurements
    """
    
    # Filter for detection objects only
    detections = annotations_gdf[annotations_gdf['objectType'] == 'detection'].copy()
    
    if len(detections) == 0:
        raise ValueError("No detections found in annotations (objectType=='detection')")
    
    # Extract precomputed centroids from measurements dict
    centroids_x = []
    centroids_y = []
    
    for idx, row in detections.iterrows():
        measurements = ast.literal_eval(row['measurements'])
        
        if 'Centroid_x' not in measurements or 'Centroid_y' not in measurements:
            raise ValueError(
                f"Row {idx} missing Centroid_x or Centroid_y in measurements dict. "
                f"Available keys: {list(measurements.keys())}"
            )
        
        centroids_x.append(measurements['Centroid_x'])
        centroids_y.append(measurements['Centroid_y'])
    
    detections['centroid_x'] = centroids_x
    detections['centroid_y'] = centroids_y
    
    return detections


def extract_tile_centroids(tiles_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract tile centroids from tile geometries (Polygons).
    
    Parameters
    ----------
    tiles_gdf : gpd.GeoDataFrame
        Tiles GeoDataFrame with Polygon geometry column
        
    Returns
    -------
    gpd.GeoDataFrame
        Copy of tiles_gdf with added columns:
        - tile_centroid_x: x-coordinate of tile centroid
        - tile_centroid_y: y-coordinate of tile centroid
    """
    
    tiles = tiles_gdf.copy()
    tiles['tile_centroid_x'] = tiles.geometry.centroid.x
    tiles['tile_centroid_y'] = tiles.geometry.centroid.y
    
    return tiles


def find_nuclei_in_tile(
    tile_geom: 'shapely.Polygon',
    tile_centroid: Tuple[float, float],
    nuclei_df: pd.DataFrame
) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float], int]:
    """
    Find nuclei within a tile and identify the closest one.
    
    For a given tile Polygon, checks which nuclei (Points) fall within the tile bounds.
    If multiple nuclei are found, returns the one closest to the tile centroid.
    
    Parameters
    ----------
    tile_geom : shapely.Polygon
        Polygon geometry of the tile
    tile_centroid : Tuple[float, float]
        (x, y) coordinates of tile centroid
    nuclei_df : pd.DataFrame
        DataFrame with columns: id, centroid_x, centroid_y, and any other nuclei info
        
    Returns
    -------
    Tuple[Optional[int], Optional[float], Optional[float], Optional[float], int]
        - closest_nuclei_id: int or None if no nuclei found
        - closest_centroid_x: float or None
        - closest_centroid_y: float or None
        - distance_to_closest: float or None (Euclidean distance in pixel space)
        - num_nuclei_in_tile: int (count of nuclei within tile bounds)
    """
    
    if len(nuclei_df) == 0:
        return None, None, None, None, 0
    
    # Create Point objects for nuclei centroids
    nuclei_points = [
        Point(row['centroid_x'], row['centroid_y'])
        for _, row in nuclei_df.iterrows()
    ]
    
    # Check which nuclei fall within the tile
    nuclei_within = []
    for idx, (nuclei_idx, row) in enumerate(nuclei_df.iterrows()):
        if tile_geom.contains(nuclei_points[idx]):
            nuclei_within.append({
                'nuclei_idx': nuclei_idx,
                'id': row['id'],
                'centroid_x': row['centroid_x'],
                'centroid_y': row['centroid_y']
            })
    
    num_nuclei = len(nuclei_within)
    
    if num_nuclei == 0:
        return None, None, None, None, 0
    
    # If only one nuclei found, return it
    if num_nuclei == 1:
        nuclei = nuclei_within[0]
        distance = np.sqrt(
            (nuclei['centroid_x'] - tile_centroid[0])**2 +
            (nuclei['centroid_y'] - tile_centroid[1])**2
        )
        return nuclei['id'], nuclei['centroid_x'], nuclei['centroid_y'], distance, num_nuclei
    
    # If multiple nuclei, find the closest to tile centroid
    closest = None
    min_distance = float('inf')
    
    for nuclei in nuclei_within:
        distance = np.sqrt(
            (nuclei['centroid_x'] - tile_centroid[0])**2 +
            (nuclei['centroid_y'] - tile_centroid[1])**2
        )
        if distance < min_distance:
            min_distance = distance
            closest = nuclei
    
    return (
        closest['id'],
        closest['centroid_x'],
        closest['centroid_y'],
        min_distance,
        num_nuclei
    )


def recenter_tile_geometry(
    nuclei_centroid: Tuple[float, float],
    tile_size_px: int = 16
) -> 'shapely.Polygon':
    """
    Create a tile Polygon geometry centered on a nuclei centroid.
    
    Generates a square Polygon of size tile_size_px × tile_size_px pixels,
    centered on the given nuclei centroid.
    
    Parameters
    ----------
    nuclei_centroid : Tuple[float, float]
        (x, y) coordinates of nuclei centroid
    tile_size_px : int, default=16
        Side length of tile in pixels
        
    Returns
    -------
    shapely.Polygon
        Rectangular Polygon representing the recentered tile
    """
    
    half_size = tile_size_px / 2.0
    x_c, y_c = nuclei_centroid
    
    # Create bounding box centered on nuclei
    return box(
        x_c - half_size,
        y_c - half_size,
        x_c + half_size,
        y_c + half_size
    )


def _process_tissue_fragment_vectorized(
    tissue_id: str,
    tiles_df: pd.DataFrame,
    nuclei_df: pd.DataFrame,
    tile_size_px: int = 16
) -> pd.DataFrame:
    """
    Process all tiles in a single tissue fragment using vectorized operations.
    
    This function is designed to be called in parallel for each tissue_id.
    
    Parameters
    ----------
    tissue_id : str
        Identifier for the tissue fragment
    tiles_df : pd.DataFrame
        Tiles data for this tissue (must have: geometry, tile_centroid_x, tile_centroid_y)
    nuclei_df : pd.DataFrame
        Nuclei data for this tissue (must have: id, centroid_x, centroid_y)
    tile_size_px : int, default=16
        Size of tiles in pixels
        
    Returns
    -------
    pd.DataFrame
        Result data with one row per tile, columns: original_geometry, recentered_geometry,
        has_nuclei, num_nuclei_in_original_tile, closest_nuclei_id, 
        closest_nuclei_centroid_x, closest_nuclei_centroid_y, distance_to_closest_nuclei
    """
    
    if len(nuclei_df) == 0:
        # No nuclei in this tissue, mark all tiles as empty
        result = pd.DataFrame({
            'original_geometry': tiles_df['geometry'].values,
            'recentered_geometry': tiles_df['geometry'].values,
            'has_nuclei': False,
            'num_nuclei_in_original_tile': 0,
            'closest_nuclei_id': None,
            'closest_nuclei_centroid_x': None,
            'closest_nuclei_centroid_y': None,
            'distance_to_closest_nuclei': None,
        })
        return result
    
    # Convert nuclei centroids to NumPy array for fast distance calculations
    nuclei_centroids = nuclei_df[['centroid_x', 'centroid_y']].values  # (N, 2)
    nuclei_ids = nuclei_df['id'].values
    
    result_data = {
        'original_geometry': [],
        'recentered_geometry': [],
        'has_nuclei': [],
        'num_nuclei_in_original_tile': [],
        'closest_nuclei_id': [],
        'closest_nuclei_centroid_x': [],
        'closest_nuclei_centroid_y': [],
        'distance_to_closest_nuclei': [],
    }
    
    # Process each tile
    for idx, tile_row in tiles_df.iterrows():
        tile_geom = tile_row['geometry']
        tile_centroid_x = tile_row['tile_centroid_x']
        tile_centroid_y = tile_row['tile_centroid_y']
        
        # Vectorized point-in-polygon check using NumPy
        nuclei_points = np.array([
            Point(x, y) for x, y in nuclei_centroids
        ])
        nuclei_in_tile_mask = np.array([tile_geom.contains(pt) for pt in nuclei_points])
        nuclei_in_tile_indices = np.where(nuclei_in_tile_mask)[0]
        
        num_nuclei = len(nuclei_in_tile_indices)
        
        result_data['original_geometry'].append(tile_geom)
        result_data['num_nuclei_in_original_tile'].append(num_nuclei)
        result_data['has_nuclei'].append(num_nuclei > 0)
        
        if num_nuclei == 0:
            # No nuclei in tile
            result_data['closest_nuclei_id'].append(None)
            result_data['closest_nuclei_centroid_x'].append(None)
            result_data['closest_nuclei_centroid_y'].append(None)
            result_data['distance_to_closest_nuclei'].append(None)
            result_data['recentered_geometry'].append(tile_geom)
        else:
            # Find closest nuclei to tile centroid using vectorized distance
            nuclei_in_tile = nuclei_centroids[nuclei_in_tile_indices]  # (M, 2)
            
            # Vectorized Euclidean distance calculation
            distances = np.sqrt(
                (nuclei_in_tile[:, 0] - tile_centroid_x)**2 +
                (nuclei_in_tile[:, 1] - tile_centroid_y)**2
            )
            
            closest_idx = np.argmin(distances)
            closest_global_idx = nuclei_in_tile_indices[closest_idx]
            closest_nuclei_id = nuclei_ids[closest_global_idx]
            closest_x = nuclei_centroids[closest_global_idx, 0]
            closest_y = nuclei_centroids[closest_global_idx, 1]
            min_distance = distances[closest_idx]
            
            result_data['closest_nuclei_id'].append(closest_nuclei_id)
            result_data['closest_nuclei_centroid_x'].append(closest_x)
            result_data['closest_nuclei_centroid_y'].append(closest_y)
            result_data['distance_to_closest_nuclei'].append(min_distance)
            
            # Recenter tile
            recentered_geom = recenter_tile_geometry(
                (closest_x, closest_y),
                tile_size_px=tile_size_px
            )
            result_data['recentered_geometry'].append(recentered_geom)
    
    return pd.DataFrame(result_data)


def align_tiles_to_nuclei(
    wsi: Dict,
    tile_size_px: int = 16,
    verbose: bool = True,
    n_jobs: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Align tile geometries to nuclei centroids for each tissue fragment (parallelized).
    
    Main processing function that:
    1. Extracts nuclei detections and their precomputed centroids
    2. For each tissue fragment, identifies nuclei within/near each tile
    3. Recenters tile geometry to focus on the closest nuclei
    4. Returns enhanced GeoDataFrame with both original and recentered geometries
    
    Uses multiprocessing to parallelize tile processing across tissue fragments.
    
    Parameters
    ----------
    wsi : Dict
        WSI (Whole-Slide Image) object containing:
        - 'tiles': gpd.GeoDataFrame with Polygon geometry
        - 'annotations': gpd.GeoDataFrame with detection/annotation objects
        Both should have 'tissue_id' column for fragment identification
        
    tile_size_px : int, default=16
        Side length of tiles in pixels (for recentering calculations)
        
    verbose : bool, default=True
        If True, print progress information
        
    n_jobs : Optional[int], default=None
        Number of parallel jobs to use. If None, uses all available CPUs.
        Set to 1 to disable parallelization (useful for debugging).
        
    Returns
    -------
    gpd.GeoDataFrame
        Enhanced GeoDataFrame with columns:
        - geometry: recentered Polygon (original if no nuclei found)
        - original_geometry: original tile Polygon
        - tissue_id: fragment identifier
        - has_nuclei: bool, True if nuclei found in original tile bounds
        - num_nuclei_in_original_tile: int, count of nuclei in original tile
        - closest_nuclei_id: int or None, ID of closest nuclei (None if empty)
        - closest_nuclei_centroid_x: float or None
        - closest_nuclei_centroid_y: float or None
        - distance_to_closest_nuclei: float or None, Euclidean distance in pixels
        - [any additional columns from original tiles_gdf]
        
    Raises
    ------
    KeyError
        If 'tiles' or 'annotations' not in wsi
    ValueError
        If no detections or invalid measurements structure
    
    Notes
    -----
    - Assumes tiles and annotations are at the same resolution (mpp)
    - Centroid coordinates should be in pixel space (not geographic)
    - Original and recentered geometries are always Polygon objects
    - Empty tiles (no nuclei found) keep original geometry but are marked with
      has_nuclei=False and null values for nuclei-related columns
    - Parallelization is per tissue_id; each tissue is processed independently
    - GeoDataFrames are not pickleable; data is converted to plain DataFrames for
      parallel processing and reconstructed afterward
    
    Examples
    --------
    >>> # Run with all available CPUs
    >>> aligned_tiles = align_tiles_to_nuclei(wsi, tile_size_px=16, n_jobs=-1)
    >>> 
    >>> # Run single-threaded for debugging
    >>> aligned_tiles = align_tiles_to_nuclei(wsi, tile_size_px=16, n_jobs=1)
    """
    
    # Validate inputs
    if 'tiles' not in wsi or 'annotations' not in wsi:
        raise KeyError("wsi must contain 'tiles' and 'annotations' keys")
    
    start_time = time.time()
    
    if verbose:
        print("Extracting nuclei detections and centroids...")
    
    # Extract and prepare data
    nuclei_gdf = extract_nuclei_centroids(wsi['annotations'])
    tiles_gdf = extract_tile_centroids(wsi['tiles'])
    
    if verbose:
        print(f"  Found {len(nuclei_gdf)} nuclei detections")
        print(f"  Processing {len(tiles_gdf)} tiles")
    
    # Get tissue IDs present in both tiles and nuclei
    tissue_ids = sorted(tiles_gdf['tissue_id'].unique())
    
    if verbose:
        print(f"  Found {len(tissue_ids)} tissue fragments")
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1
    
    if verbose:
        print(f"  Using {n_jobs} parallel jobs")
    
    # Prepare data for parallel processing
    # Convert GeoDataFrames to regular DataFrames for pickling
    tiles_df = pd.DataFrame(tiles_gdf)
    nuclei_df = pd.DataFrame(nuclei_gdf)
    
    # Create partial function with fixed parameters
    process_fn = partial(
        _process_tissue_fragment_vectorized,
        tile_size_px=tile_size_px
    )
    
    # Prepare tasks: (tissue_id, tiles_for_tissue, nuclei_for_tissue)
    tasks = []
    for tissue_id in tissue_ids:
        tiles_for_tissue = tiles_df[tiles_df['tissue_id'] == tissue_id].reset_index(drop=True)
        nuclei_for_tissue = nuclei_df[nuclei_df['tissue_id'] == tissue_id].reset_index(drop=True)
        tasks.append((tissue_id, tiles_for_tissue, nuclei_for_tissue))
    
    # Process tasks in parallel
    if n_jobs == 1:
        # Single-threaded mode (useful for debugging)
        results = []
        for tissue_id, tiles_for_tissue, nuclei_for_tissue in tasks:
            result = process_fn(tissue_id, tiles_for_tissue, nuclei_for_tissue)
            results.append(result)
    else:
        # Multi-threaded mode
        with Pool(n_jobs) as pool:
            results = pool.starmap(process_fn, tasks)
    
    if verbose:
        print("Combining results...")
    
    # Concatenate results from all tissue fragments
    all_results = pd.concat(results, ignore_index=True)
    
    # Create result GeoDataFrame with recentered geometry as primary
    result_gdf = tiles_gdf.copy()
    result_gdf['geometry'] = all_results['recentered_geometry'].values
    result_gdf['original_geometry'] = all_results['original_geometry'].values
    result_gdf['has_nuclei'] = all_results['has_nuclei'].values
    result_gdf['num_nuclei_in_original_tile'] = all_results['num_nuclei_in_original_tile'].values
    result_gdf['closest_nuclei_id'] = all_results['closest_nuclei_id'].values
    result_gdf['closest_nuclei_centroid_x'] = all_results['closest_nuclei_centroid_x'].values
    result_gdf['closest_nuclei_centroid_y'] = all_results['closest_nuclei_centroid_y'].values
    result_gdf['distance_to_closest_nuclei'] = all_results['distance_to_closest_nuclei'].values
    
    # Clean up intermediate columns
    cols_to_drop = ['tile_centroid_x', 'tile_centroid_y']
    result_gdf = result_gdf.drop(columns=cols_to_drop)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        total_recentered = result_gdf['has_nuclei'].sum()
        total_empty = (~result_gdf['has_nuclei']).sum()
        print(f"  Recentered: {total_recentered} tiles")
        print(f"  Empty tiles (no nuclei): {total_empty}")
        print(f"Done! Elapsed time: {elapsed_time:.2f}s")
    
    return result_gdf
