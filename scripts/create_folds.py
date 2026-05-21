import warnings
import random
import os
import pandas as pd
import yaml
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    with open('/path/to/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_test', type=bool, default=True, help='Wether to create a test or just val/train splits. Default is True (i.e. no test set)')
    args = parser.parse_args()


    df_path_ucl = config['data_splits']['df_path_UCL']
    df_path_newcastle = config['data_splits']['df_path_Newcastle']
    n_folds = config['data_splits']['n_folds']
    folds_dir = config['data_splits']['folds_dir']
    split_ratio = config['data_splits']['split_ratio']
    features_dir = config['data_splits']['features_dir']
    n_cases = config['data_splits']['n_cases']
    
    if not os.path.exists(folds_dir):
        os.makedirs(folds_dir)

    # take this number of patients per cohort
    n_patients = n_cases

    # ---- Process each cohort and collect into a list ----
    cohort_dfs = []

    for cohort_path in [df_path_ucl, df_path_newcastle]:
        df = pd.read_csv(cohort_path)

        if "Newcastle" in cohort_path:
            cohort_label = "Newcastle"
            # Filter to CD and UC only (no HC in Newcastle)
            df = df[df['Diagnosis_DAA'].isin(['CD', 'UC'])].copy()
            # Derive donor_id by stripping .mrxs suffix
            df['Slide_name'] = df['Image_name'].str.replace('.mrxs', '', regex=False)
            # Normalise condition column
            df['Condition'] = df['Diagnosis_DAA']
            # Patient-level identifier
            df['patient_id'] = df['Release_ID'].astype(str)
        else:
            cohort_label = "UCL"
            # Filter to CD and UC only (no HC in UCL for now)
            df = df[df['patient_diagnosis_majority'].isin(['CD', 'UC'])].copy()
            # Derive donor_id by stripping .ndpi suffix
            df['Slide_name'] = df['pID'].str.replace('.ndpi', '', regex=False)
            # Normalise condition column
            df['Condition'] = df['patient_diagnosis_majority']
            # Patient-level identifier
            df['patient_id'] = df['p_case'].astype(str) + '_case'

        df['Cohort'] = cohort_label

        # Sample up to n_patients unique patients per condition to ensure class balance
        sampled_patients_list = []
        for condition, grp in df.groupby('Condition'):
            unique_pts = grp['patient_id'].unique()
            n_to_sample = min(n_patients, len(unique_pts))
            sampled = pd.Series(unique_pts).sample(n=n_to_sample, random_state=42).values
            sampled_patients_list.append(sampled)
            print(f"[{cohort_label}] {condition}: {len(unique_pts)} total patients, sampled {n_to_sample}")

        df = df[df['patient_id'].isin(np.concatenate(sampled_patients_list))]

        # Keep only the columns we need
        df = df[['Slide_name', 'Condition', 'patient_id', 'Cohort', 'full_path']].reset_index(drop=True)
        df = df.rename(columns={'full_path': 'Slide'})

        df['Feature_Path'] = df['Slide'].apply(
            lambda x: os.path.join(features_dir, f"{os.path.basename(x).split('.')[0]}.h5")
        )
        cohort_dfs.append(df)

        print(f"[{cohort_label}] total slides after balanced sampling: {len(df)}")

    # Concatenate both cohorts
    df_all = pd.concat(cohort_dfs, ignore_index=True)

    # ---- Patient-level splitting to avoid data leakage ----
    # Build a patient-level table with majority condition per patient
    patient_df = (
        df_all.groupby('patient_id')
        .agg(Condition=('Condition', 'first'), Cohort=('Cohort', 'first'))
        .reset_index()
    )
    # Joint stratification key: condition x cohort
    patient_df['strat_key'] = patient_df['Condition'] + '_' + patient_df['Cohort']
    patients = patient_df['patient_id'].values
    strat_keys = patient_df['strat_key'].values

    for fold_idx in range(5):
        # 70 / 15 / 15 split at the patient level
        train_patients, temp_patients, train_strat, temp_strat = train_test_split(
            patients, strat_keys,
            test_size=split_ratio[1], stratify=strat_keys, random_state=42 + fold_idx
        )
        val_patients, test_patients, _, _ = train_test_split(
            temp_patients, temp_strat,
            test_size=0.50, stratify=temp_strat, random_state=42 + fold_idx
        )

        # Map patient-level splits back to all slides
        train_set = set(train_patients)
        val_set = set(val_patients)
        if args.no_test:
            # merge val and test sets into a single val set if no_test flag is True
            val_set = val_set.union(set(test_patients))

        out_df = df_all.copy()
        out_df['split'] = out_df['patient_id'].apply(
            lambda p: 'train' if p in train_set
            else ('val' if p in val_set else 'test')
        )

        out_df = out_df[['Slide_name', 'Condition', 'split', 'patient_id', 'Cohort', 'Slide', 'Feature_Path']]
        out_df.to_csv(f'{folds_dir}/fold_{fold_idx}.csv', index=False)

        # Print split summary for this fold
        split_summary = out_df.groupby(['split', 'Cohort', 'Condition']).size().reset_index(name='n_slides')
        print(f"\n--- Fold {fold_idx} ---")
        print(split_summary.to_string(index=False))