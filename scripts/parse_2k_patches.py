# This code is used to parse 2k image patches for IBD predictive modeling.
# Code is adapted from: https://github.com/AstraZeneca/ibd-interpret/blob/master/HIPT/parse_4k.py

"""
Create 256x256 patches from 2048x2048 patches
"""
from tracemalloc import start
from PIL import Image
import numpy as np
import glob
import argparse
import random
import os
import time
import pandas as pd
import time

def filter_empty_patches(patches, background_threshold=190, background_fraction=0.95):
    """
    Filter out patches without tissue based on RGB thresholding.
    
    Args:
        patches: numpy array of shape (N, H, W, C)
        background_threshold: RGB mean threshold for background detection (default 190)
        background_fraction: fraction of background pixels allowed (default 0.95)
    
    Returns:
        filtered_patches: numpy array with only tissue-containing patches
        tissue_mask: boolean array indicating which patches contain tissue
    """
    tissue_mask = []
    
    for i in range(patches.shape[0]):
        patch = patches[i]
        # Compute mean across RGB channels for each pixel
        rgb_mean = np.mean(patch, axis=2)
        # print(rgb_mean.shape)
        # Mark pixels as background where mean > threshold
        background_pixels = rgb_mean > background_threshold
        # Calculate fraction of background pixels
        bg_fraction = np.sum(background_pixels) / (patch.shape[0] * patch.shape[1])
        # Keep patch if background fraction is below threshold
        tissue_mask.append(bg_fraction < background_fraction)
    
    tissue_mask = np.array(tissue_mask)
    filtered_patches = patches[tissue_mask]
    
    return filtered_patches, tissue_mask

def parse_2k(region_path, background_threshold=190, background_fraction=0.95):

    # load in region and convert to np 
    # the region will never be perfectly 2048x2048 because of the floating resolution mpp.
    # to account for this, we will keep patches of 256x256 and extract 64 patches. 
    # I will remove the excess pixel both from left/right and up/down
    region = np.array(Image.open(region_path))

    # get real pixel dimensions of the region
    true_dim = region.shape[0]
    excess = true_dim - 2048
    offset = excess // 2

    # define a tensor to hold the patches
    patches_256 = np.zeros((64,256,256,3))

    # counter counts from 0 to 63 for indexing
    counter = 0
    for i in range(8):
        # for each row (8 rows)
        # row index multiplied by 256 to get patch coordinate
        ix = offset + (i * 256)
        
        for j in range(8):
            # for each column (8 columns)
            # column index multiplied by 256 to get patch coordinate
            jx = offset + (j * 256)
            # get patch
            patch = region[ix:ix+256, jx:jx+256, :]
            patches_256[counter] = patch
            counter+=1

    # Filter out empty patches without tissue
    filtered_patches, tissue_mask = filter_empty_patches(
        patches_256,
        background_threshold=background_threshold,
        background_fraction=background_fraction
    )
    
    return filtered_patches, tissue_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", type=str, default=None, help="path to 2k patches")
    parser.add_argument("--save_dir", type=str, default=None, help="where to save 256x256 patches")
    parser.add_argument("--background_threshold", type=int, default=190, help="RGB mean threshold for background detection")
    parser.add_argument("--background_fraction", type=float, default=0.95, help="fraction of background pixels allowed")

    args = parser.parse_args()

    bag_list = os.listdir(args.patch_dir)

    random.shuffle(bag_list)

    for i, bag in enumerate(bag_list):

        print(f"parsing patches for: {bag}")        
        region_paths = glob.glob(os.path.join(args.patch_dir, bag, "*.png"))
        # Remove png used to check the tiling process for the whole WSI
        region_paths = [path for path in region_paths if "wholeWSI" not in path] 

        print(f"number of 2k regions to parse: {len(region_paths)}")
        start = time.time()

        for region_path in region_paths:
            try:
                filtered_patches, tissue_mask = parse_2k(
                    region_path,
                    background_threshold=args.background_threshold,
                    background_fraction=args.background_fraction
                )
            except Exception as e:
                print(e)
                print("could not parse patches for ", region_path)
                continue

            # Print statistics
            total_patches = len(tissue_mask)
            kept_patches = np.sum(tissue_mask)
            filtered_patches_count = total_patches - kept_patches
            print("Patch filtering statistics:")
            print(f"  Total patches extracted: {total_patches}")
            print(f"  Patches with tissue (kept): {kept_patches} ({kept_patches/total_patches*100:.1f}%)")
            print(f"  Empty patches (filtered): {filtered_patches_count} ({filtered_patches_count/total_patches*100:.1f}%)")
            # Save only patches with tissue
            for j in range(len(filtered_patches)):
                patch_256 = Image.fromarray(filtered_patches[j,:,:,:].astype(np.uint8)).convert('RGB')
                patch_256.save( os.path.join(args.save_dir, bag + "_" + region_path.split("/")[-1].split(".")[0] + f"_patch_{str(j)}.png" ) )

            print("done")

        end = time.time()

        print("done")
        print(f"completed parsing for {i+1} / {len(bag_list)}, progress: { ((i+1)/len(bag_list)) * 100}, took {end-start} seconds")

if __name__ == '__main__':
    main()