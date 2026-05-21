import os
import pandas as pd
import argparse

# this script is to prepare the final data for training by creating a column
# in the fold csv files with a column that says whether or not to use that slide
# this is because some slides might fail during preprocessing and feature extraction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_dir', type=str, help='full path to the directory containing the fold csv files')
    args = parser.parse_args()

    for fold_idx in range(5):
        fold_path = f"{args.fold_dir}/fold_{fold_idx}.csv"
        if not os.path.exists(fold_path):
            print(f"[fold_{fold_idx}] CSV not found, skipping: {fold_path}")
            continue

        df = pd.read_csv(fold_path)

        df['use_slide'] = df['Feature_Path'].apply(
            lambda p: os.path.isfile(p) and os.path.getsize(p) > 0
        )

        excluded = df.loc[~df['use_slide'], 'Slide_name'].tolist()
        print(f"\n[fold_{fold_idx}] {len(excluded)} slide(s) excluded (missing or empty .h5):")
        for slide in excluded:
            print(f"  - {slide}")
        if not excluded:
            print("  None — all slides have valid .h5 files.")

        df.to_csv(fold_path, index=False)
        print(f"[fold_{fold_idx}] Updated CSV saved: {fold_path}")

    