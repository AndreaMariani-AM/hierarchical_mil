import sys
import os
from PIL import Image
from models.HIPT_2K import HIPT_2K
from utils.hipt_utils import eval_transforms
import torch
import numpy as np
import argparse
import glob
import random


# function for generating embedding for a 2k patch
def generate_embedding(region, model):

    x = eval_transforms()(region).unsqueeze(dim=0)
    out = model.generate_vit256_embeddings(x)

    embedding = out.cpu().numpy()

    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default=None, help="path to where patches are")
    parser.add_argument('--feats_dir', type=str, default=None, help='Path to where embeddings will be stored')
    parser.add_argument('--checkpoint', type=str, default='/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/experiments/training/HIPT/256_ssl/checkpoint0080.pth', help="checkpoint to use, defaults to authors")
    parser.add_argument('--task_id', type=int, default=None, help="Task ID for distributed processing (from SLURM_ARRAY_TASK_ID)")
    parser.add_argument('--num_tasks', type=int, default=None, help="Total number of tasks")
    args = parser.parse_args()

    # get the list of bags
    bag_list = glob.glob( os.path.join(args.patch_dir, '*') )
    
    # Filter out the embeddings folder
    bag_list = [bag for bag in bag_list if "extracted" not in bag]
    bag_list.sort()  # Sort for reproducibility across tasks

    print("total WSIs found:", len(bag_list))

    # Distribute bags across tasks if running in array job mode
    if args.task_id is not None and args.num_tasks is not None:
        bags_per_task = len(bag_list) // args.num_tasks
        start_idx = args.task_id * bags_per_task
        
        # Last task handles any remainder
        if args.task_id == args.num_tasks - 1:
            end_idx = len(bag_list)
        else:
            end_idx = start_idx + bags_per_task
        
        bag_list = bag_list[start_idx:end_idx]
        print(f"Task {args.task_id}/{args.num_tasks}: Processing {len(bag_list)} WSIs (indices {start_idx} to {end_idx-1})")
    else:
        # If not running as array job, shuffle as before
        random.shuffle(bag_list)
        print("Running in single-task mode")

    # model init
    model = HIPT_2K(model256_path=args.checkpoint, device256='cuda:0')
    model.eval()

    if not os.path.exists(args.feats_dir):
        os.makedirs(args.feats_dir, exist_ok=True)

    for bag_idx, bag in enumerate(bag_list):

        # get the list of patches for this bag
        regions = glob.glob( os.path.join(bag, "*") )

        regions_filt = [path for path in regions if "wholeWSI" not in path] 

        print(f"[{bag_idx+1}/{len(bag_list)}] processing {bag}, num regions: {len(regions_filt)}")

        for region_path in regions_filt:
            feat_path = os.path.join(
                args.feats_dir,
                os.path.splitext(os.path.basename(bag))[0] + "_" +
                os.path.splitext(os.path.basename(region_path))[0] + ".pt",
            )

            if os.path.exists(feat_path):
                # print("skipping, features already generated")  # Commented to reduce log spam
                continue

            try:
                # try to open patch
                region = Image.open(region_path).convert('RGB')
            except Exception as e:
                # log
                print("region could not be loaded:", region_path)
                print(e)
                continue

            patch_256_embeddings = generate_embedding(region, model)

            # save embeddings as a pt file
            torch.save(torch.from_numpy(patch_256_embeddings), feat_path)

    print(f"Task completed. Processed {len(bag_list)} bags.")


if __name__ == '__main__':
    main()