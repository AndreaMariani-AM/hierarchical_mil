import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))  # to avoid OpenCV DecompressionBombError for large images
import sys
import lazyslide as ls
import argparse
import pandas as pd
import yaml
import h5py
import numpy as np
from pathlib import Path
from wsidata import open_wsi
import matplotlib.pyplot as plt
import time
import torch
import scanpy as sc
from typing import Any, Dict, List
import src.data.preprocessing as pp
from src.models.ViT import UNI2Dense, Virchow2Dense
from src.utils.experts_utils import get_transform


# The script takes in a slide, and performs three major steps: 
#   1) Segmentation, 
#   2) Tile Extraction
#   3) Feature Extraction using a pretrained model (e.g., Virchow2)
# In the future these steps may be decoupled into separate scripts for modularity.

parser = argparse.ArgumentParser()
# Primary argument: slide index for SLURM array task
parser.add_argument('--idx', type=int, default=None, help='slide index for SLURM array task')
parser.add_argument('--config_file', type=str, default=None, help='Path to config file')
args = parser.parse_args()

with open(args.config_file, "r") as file_yml:
    config_file = yaml.safe_load(file_yml)

# get device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}', flush=True)

def segmentation(wsi: Any, config_file: Dict[str, Any], slide_name: str) -> None:
    
    model_segmentation = config_file['Segmentation']['model_type']
    MPP = config_file['Segmentation']['mpp']
    outdir_seg = config_file['Segmentation']['outdir_segmentation']

    os.makedirs(outdir_seg, exist_ok=True)
    
    # Segmentation
    ls.seg.tissue(wsi, key_added="tissues", model=model_segmentation, mpp=MPP*5, device='cpu') # mpp=2.5 is the correct resolution for PathProfiler
    
    # get peak memory after segmentation
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory during Segmentation: {peak_memory:.2f}GB")

    # save segmentation mask as H5
    seg_path = os.path.join(outdir_seg, f"{slide_name}_segmentation_mask.h5")
    wkt_strings = [geom.wkt for geom in wsi['tissues'].geometry]
    with h5py.File(seg_path, 'w') as h5f:
        h5f.create_dataset('segmentation/geometry_wkt', data=wkt_strings, dtype=h5py.string_dtype())

    # save segmentation visualization to double check quality of segmentation
    masks_dir = os.path.join(outdir_seg, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    ls.pl.tissue(wsi, tissue_key="tissues", ax=ax)
    fig.savefig(os.path.join(masks_dir, f"{slide_name}_mask.png"))
    plt.close(fig)

    print(f'Segmentation mask saved!!: {slide_name}')

def tiling(wsi: Any, config_file: Dict[str, Any], slide_name: str, cohort: str = "UCL") -> None:
    # Custom tiling to big region + method to extract sub patches for each
    # from here: /group/glastonbury/andrea/projects/IBD/IBD_predictive_model/scripts/create_2K_patches.py
    
    # Tiling
    tile_size = config_file['Tiling']['tile_size']
    region_size = config_file['Tiling']['region_level_size']
    tissue_threshold = config_file['Tiling']['tissue_threshold']
    tiling_MPP = config_file['Tiling']['mpp']
    # outdir_tiles = config_file['Tiling']['outdir_tiling']

    # if not os.path.exists(outdir_tiles):
    #     os.makedirs(outdir_tiles)

    # Validate and repair geometries before tiling
    tissue_fragments = wsi['tissues']
    invalid_count = (~tissue_fragments.geometry.is_valid).sum()
    
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid geometries. Repairing...", flush=True)
        tissue_fragments.geometry = tissue_fragments.geometry.buffer(0)
        wsi['tissues'] = tissue_fragments
        print(f"Geometries repaired successfully", flush=True)
    else:
        print(f"All geometries are valid", flush=True)

    # Newcastle slides are faded — RGB background filtering is too aggressive on their channels.
    # For Newcastle, skip it (background_fraction=1.0) and rely solely on filter_tiles_by_tissue_mask.
    # For all other cohorts (UCL) use the configured tissue_threshold.
    if cohort == "Newcastle":
        bg_fraction = 1.0
        print(f"[INFO] Newcastle slide — using background_fraction=1.0 (tissue mask filtering only)", flush=True)
    else:
        bg_fraction = tissue_threshold
        print(f"[INFO] UCL slide — using background_fraction={bg_fraction}", flush=True)

    pp.tile_whole_wsi_with_background_filter(
        wsi,
        tile_px=region_size,
        mpp=tiling_MPP,
        background_fraction=bg_fraction,
        key_added="region_tiles",
        return_tiles=False
    )
    # Use segmentation mask to keep only tiles that overlap with tissue regions
    pp.filter_tiles_by_tissue_mask(wsi, tile_key="region_tiles", tissue_key="tissues")

    # Create sub-tiles from the big tiles that overlap with tissue regions
    pp.create_subtiles_from_region_tiles(
        wsi,
        region_tile_key="region_tiles",
        subtile_px=tile_size,
        mpp=tiling_MPP,
        background_fraction=1.0, # skip filtering on background as it is too coarse for subtiles
        key_added="tiles",
        return_tiles=False,
        edge=True
    )
    # Use tissue polygons for accurate sub-tile filtering instead
    pp.filter_tiles_by_tissue_mask(wsi, tile_key="tiles", tissue_key="tissues")
    
    # Re-index to contiguous IDs so region_tile_id is a valid
    # positional index into /regions/ in the output H5 file.
    pp.reindex_tiles_and_regions(
        wsi,
        region_tile_key="region_tiles",
        subtile_key="tiles",
    )

    # We need to remove Newcastle artifacts by predicting tile level artifacts
    if cohort == "Newcastle":
        print(f"[INFO] Newcastle slide — predicting tile-level artifacts and filtering tiles with >60% artifact content", flush=True)
        ls.tl.tile_prediction(wsi, model="pathprofilerqc", tile_key="tiles")
        
        # filter tissue_ID with high artifact content, > 60%
        tissue_to_keep = wsi['tiles'].groupby('tissue_id')['misc_artifacts_present'].median() <= 0.6

        # removing tiles from dropped tissues
        tmp =  wsi['tissues'][wsi['tissues'].tissue_id.isin(tissue_to_keep[tissue_to_keep].index)]
        if not tmp.empty:
            wsi['tissues'] = tmp
        else:
            print(f"WARNING: No tissues left after artifact filtering for slide {slide_name}. Do not use this slide", flush=True)
            sys.exit(1)
        wsi['region_tiles'] = wsi['region_tiles'][wsi['region_tiles'].tissue_id.isin(wsi['tissues'].tissue_id)]
        wsi['tiles'] = wsi['tiles'][wsi['tiles'].tissue_id.isin(wsi['tissues'].tissue_id)]

        # reindex the tiles
        pp.reindex_tiles_and_regions(
            wsi,
            region_tile_key="region_tiles",
            subtile_key="tiles",
        )

def save_h5_metadata(
        wsi: Any, 
        config_file: Dict[str, Any], 
        slide_name: str
) -> str:
    """
    Interface with saving function for H5 metadata
    """

    outdir_features = config_file['Feature_Extraction']['outdir_features']
    model_name = config_file['Feature_Extraction']['model_type']
    tile_size = config_file['Tiling']['tile_size']
    emb_dim=config_file['Feature_Extraction']['embedding_dim']
    
    # Save H5 structure
    pp.save_tile_region_h5(wsi, outdir_features, model_name, tile_size, slide_name, embd_dim=emb_dim, pool_cell_tokens=True)

def extract_features(wsi: Any, config_file: Dict[str, Any], slide_name: str) -> None:
    """Run feature extraction on sub-tiles. Embeddings stay in wsi object."""
    model_feature_extraction = config_file['Feature_Extraction']['model_type']
    batch_size = config_file['Feature_Extraction']['batch_size']

    # reset peak memory before feature extraction
    torch.cuda.reset_peak_memory_stats()

    if model_feature_extraction == "virchow2":
        m = Virchow2Dense()
        model_name = "virchow2"
    if model_feature_extraction == "UNI":
        m = UNI2Dense()
        model_name = "UNI"

    # Extraction
    region, patch, cell = pp.feature_extraction(
        wsi,
        model=m,
        model_name=model_name,
        transform=get_transform(),
        device=DEVICE,
        batch_size=batch_size,
        pbar=False,
        return_features=True,
        pool_cell_tokens=True,  # 4×4 avg pool on GPU: (N,256,1280) → (N,16,1280) # 16 tokens at 32um each
    )
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory during Feature Extraction: {peak_memory:.2f}GB")
    print("\n" * 2)
    print(f'Checking the correct feature shapes for region, patch, and cell features...', flush=True)
    print(f"Region feature shape should be (N, 2560) and is: {region.shape}", flush=True)
    print(f"Patch feature shape should be (N, 1280) and is: {patch.shape}", flush=True)
    print(f"Cell feature shape should be (N, 16, 1280) and is: {cell.shape}", flush=True)

    pp.save_embeddings_to_h5(wsi, model_feature_extraction, region, patch, cell)


def compute_low_dim(wsi, config_file, slide_name) -> None:
    # compute low dim representation and clustering
    model_emb = config_file['Feature_Extraction']['model_type']
    outdir_features = config_file['Feature_Extraction']['low_dim']
    tile_size = config_file['Tiling']['tile_size']

    os.makedirs(os.path.join(outdir_features,  f"{model_emb}_{tile_size}"), exist_ok=True)

    # Configure scanpy to save figures to the correct directory
    sc.settings.figdir = os.path.join(outdir_features, f"{model_emb}_{tile_size}")
    
    adata = wsi[f"{model_emb}_tiles"]
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", resolution=0.2)
    sc.pl.umap(adata, color="leiden", save=f'_{slide_name}_umap.png')

    # save anndata to a parent directory
    os.makedirs(os.path.join(outdir_features, 'h5ad'), exist_ok=True)
    adata.write(os.path.join(outdir_features, 'h5ad', f"{slide_name}_anndata.h5ad"))

    # map it back to the wsi object
    fig, ax = plt.subplots()
    ls.pl.tiles(
        wsi,
        feature_key=f"{model_emb}_tiles",
        color="leiden",
        alpha=0.5,
        palette=adata.uns["leiden_colors"],
        show_contours=False,
        ax=ax
        )
    fig.savefig(os.path.join(outdir_features, f"{model_emb}_{tile_size}", f"{slide_name}_leiden_clusters.png"), dpi=400)

def extract_features_from_slide(chunk: List[str], config_file: Dict[str, Any], cohort: str = "UCL") -> None:
    print('Job started...')
    slide = chunk[0]
    
    # read slide
    wsi = open_wsi(slide)
    # extract slide name from the path
    slide_name = Path(wsi._reader.file).stem

    # Segment the tissue
    segmentation(wsi, config_file, slide_name)
    # Tile the tissue
    tiling(wsi, config_file, slide_name, cohort=cohort)

    # Save tile & region metadata to H5 (before extraction)
    save_h5_metadata(wsi, config_file, slide_name)
    # Free the region-tile slot — no longer needed
    del wsi.shapes['region_tiles']
    print('Deleted region_tiles to avoid FM extraction issues', flush=True)

    # Extract features from tiles at multiple levels and save to wsi object
    extract_features(wsi, config_file, slide_name)

    # Compute low dim representation and clustering
    compute_low_dim(wsi, config_file, slide_name)


if __name__ == '__main__':
    t = time.time()
    
    # reset cuda memory stats
    torch.cuda.reset_peak_memory_stats()

    # Slide directory
    slides_dir = config_file['Segmentation']['slide_dir']

    # Process all the slides in csv file
    # skip header if present
    slides_df = pd.read_csv(slides_dir, sep=',', header=0)
    slides = slides_df['Slide'].tolist()
    cohorts = slides_df['Cohort'].tolist()

    print('Number of files:', len(slides), flush=True)

    # Execution mode: idx-based, one slide per task
    if args.idx is not None:
        # === IDX MODE (SLURM array - one slide per task) ===
        print('Running in idx mode (SLURM array, one slide per task)', flush=True)
        idx = int(args.idx)
        
        # Validate index bounds
        if idx >= len(slides):
            print(f'ERROR: Index {idx} out of bounds. Total slides: {len(slides)}', flush=True)
            sys.exit(1)
        
        # Process single slide
        chunk = [slides[idx]]
        cohort = cohorts[idx]
        print(f'IDX mode: Processing slide {idx}: {os.path.basename(slides[idx])} (Cohort: {cohort})', flush=True)
        
        extract_features_from_slide(chunk, config_file, cohort=cohort)
        

    print('Tissue feature extraction done (%.2f)s' % (time.time() - t))