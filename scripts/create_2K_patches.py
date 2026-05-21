# Use the segmentation done for ADMIL model to extract 4k patches only for tissue regions. Save a png for each patch with the naming convention:
# <slide_id>_x<x_coord>_y<y_coord>_4096.png
import time
import os
import sys
import src.data.preprocessing as pp
import lazyslide as ls
from wsidata import open_wsi
from pathlib import Path
import argparse
import yaml
import pandas as pd
from typing import Any, Dict, List
import matplotlib.pyplot as plt

# Create 2048 patches from segmentation masks

parser = argparse.ArgumentParser()
# Primary argument: slide index for SLURM array task
parser.add_argument('--idx', type=int, default=None, help='slide index for SLURM array task')
parser.add_argument('--config_file', type=str, default=None, help='Path to config file')
args = parser.parse_args()

with open(args.config_file, "r") as file_yml:
    config_file = yaml.safe_load(file_yml)

def patch_tissues(chunk: List[str], config_file: Dict[str, Any]) -> None:
    """
    Patch tissues from segmentation masks.
    
    :param chunk: Description
    :type chunk: List[str]
    :param config_file: Description
    :type config_file: Dict[str, Any]
    """
    print("Patching tissues...", flush=True)
    sld=chunk[0]

    outdir_imgs = config_file['Patching']['outdir_imgs']
    patch_size = config_file['Patching']['patch_size']
    segmentation_mask_dir = config_file['Patching']['segmentation_masks']
    mpp = config_file['Patching']['mpp']

    # read slide
    wsi = open_wsi(sld)
    # extract slide name from the path
    slide_name = Path(wsi._reader.file).stem

    final_outdir = os.path.join(outdir_imgs, slide_name)

    if not os.path.exists(final_outdir):
        os.makedirs(final_outdir)

    print(f"Processing slide: {slide_name}", flush=True)

    # First tiling with background filtering
    pp.tile_whole_wsi_with_background_filter(
    wsi,
    tile_px=patch_size,
    mpp=mpp,
    background_fraction=0.95,  # Keep tiles with <95% background, to allow small tissues to be included
    key_added="tiles",
    return_tiles=False)

    # Load segmentation mask
    seg_mask_file = os.path.join(segmentation_mask_dir, f"{slide_name}_segmentation_mask.geojson")
    ls.io.load_annotations(wsi, annotations=seg_mask_file, key_added='tissues')

    # Use segmentation mask to keep only tiles that overlap with tissue regions
    pp.filter_tiles_by_tissue_mask(wsi)
    
    # Save patches
    pp.extract_tiles_from_wsi(wsi, output_dir=final_outdir, target_mpp=mpp)

    # save an image to see if patching works
    fig, ax = plt.subplots(figsize=(10, 10))
    ls.pl.tiles(wsi, linewidth=0.8, ax=ax)
    fig.savefig(os.path.join(final_outdir, f"{slide_name}_wholeWSI.png"))

if __name__ == "__main__":
    print("Creating 2k patches from segmentation masks")
    t = time.time()

    # Slide directory
    slides_dir = config_file['Patching']['slide_dir']

    # Process all the slides in csv file
    # skip header if present
    slides = pd.read_csv(slides_dir, sep = ',', header=0)['Slide'].tolist()

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
        print(f'IDX mode: Processing slide {idx}: {os.path.basename(slides[idx])}', flush=True)
        
        patch_tissues(chunk, config_file)
        

    print('Tissue tissue patching done (%.2f)s' % (time.time() - t))
