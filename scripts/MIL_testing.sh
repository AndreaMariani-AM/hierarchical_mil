#!/bin/bash

module load mpi
module load cuda11.7
module load cudnn8.5-cuda11.7

# eval "$(mamba shell hook --shell bash)"
# mamba activate /group/glastonbury/conda_envs/lazyslide.v0.9.3

# Set config files
config_file="../configs/train_config.yaml"


# Run the Python script for this fold
python ../scripts/train_MIL.py \
--config_file $config_file \
--fold 0 \
--model_type "hierarchical" \
--input_dim 2560 \
--pooling "sum" \
--frozen_backbone "Virchow2" \
--scale_drop_p 0.15 \
--max_epochs 500 \
--accumulate_grad_batches 12 \
--cell_warmup_start 0 \
--cell_warmup_epochs 0 \
--loss "focal" \
--lr 1e-4 \
--weight_decay 1e-3 \
--top_k 512 \
--patient_level \
--comment "patient_aware_training"