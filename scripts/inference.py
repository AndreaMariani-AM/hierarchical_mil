import os
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import time
import yaml
import argparse
import pandas as pd
from src.data.dataset import (
    RepresentationsDataset,
    HierarchicalRepresentationsDataset,
    hierarchical_collate_fn,
)
from torch.utils.data import DataLoader
import torch
import lightning as L
from src.training.trainer import MILTrainer
from lightning.pytorch.callbacks import ModelCheckpoint

torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)  # if using multi-GPU

# model definition
# inference loop

if __name__ == '__main__':
    """
    Main inference script.
    """
    t = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Path to config file')
    parser.add_argument('--check_path', type=str, default=None, help='Path to checkpoint file')
    # parser.add_argument('--fold', type=str, default=None, help='Fold number')
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['attention', 'additive', 'hierarchical'],
                        help='MIL model type: "attention" or "additive" or "hierarchical" (default: attention)')
    # parser.add_argument('--pooling', type=str, default='sum',
    #                     choices=['sum', 'mean', 'max'],
    #                     help='Pooling strategy for AdditiveMIL (default: sum)')
    # parser.add_argument('--frozen_backbone', type=str, default='Virchow2',
    #                     choices=['Virchow2', 'UNI2'],
    #                     help='Frozen backbone to extract dense patch tokens')
    # parser.add_argument('--top_k', type=int, default=128, 
    #                     help='Top k attended patches to use for Cell Expert in HierarchicalMIL (default: 128)')
    # parser.add_argument('--scale_drop_p', type=float, default=0.15, 
    #                     help='Probability of dropping a scale in HierarchicalMIL (default: 0.15)')
    # parser.add_argument('--max_epochs', type=int, default=100, 
    #                     help='Maximum number of training epochs (default: 100)')
    # parser.add_argument('--accumulate_grad_batches', type=int, default=32, 
    #                     help='Number of batches to accumulate gradients over (default: 32)')
    # parser.add_argument('--lr', type=float, default=1e-4,
    #                     help='Learning rate for the optimizer (default: 1e-4)')
    # parser.add_argument('--weight_decay', type=float, default=1e-3,
    #                     help='Weight decay for the optimizer (default: 1e-3)')
    # parser.add_argument('--hidden_dim', type=int, default=1280,
    #                     help='Hidden dimension size for the MIL model (default: 1280)')
    # parser.add_argument('--hidden_dim_2', type=int, default=256,
    #                     help='Common final dimension for all MIL Experts (default: 256)')
    # parser.add_argument('--dropout', type=float, default=0.25,
    #                     help='Dropout rate for the MIL model (default: 0.25)')
    # parser.add_argument('--cell_warmup_start', type=int, default=20,
    #                     help='Epoch to activate CellExpert (default: 20)')
    # parser.add_argument('--cell_warmup_epochs', type=int, default=10,
    #                     help='Epoch to warmup CellExpert (default: 10)')
    # parser.add_argument('--cell_warmup_lr_factor', type=int, default=0.1,
    #                     help='LR multiplier for CellExpert during warm-up (default: 0.1)')
    
    args = parser.parse_args()

    # read config file
    with open(args.config_file, "r") as file_yml:
        config_file = yaml.safe_load(file_yml)

    folds_dir = Path(config_file['training']['folds_dir'])
    hierarchical_h5_dir = Path(config_file['training']['hierarchical_h5_dir'])
    outdir = Path(config_file['training']['out_dir'])
    
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    
    # For later when training is on only one split the fold wouldn't matter, but for now just use fold 0; can be parameterized later
    fold_num=0
    # fold_num=0# For now, just use fold 0; can be parameterized later
    train_split_file = folds_dir / f'fold_{fold_num}.csv'

    # create dataset and dataloader
    if args.model_type == 'hierarchical':
        predict_dataset = HierarchicalRepresentationsDataset(
            csv_path=train_split_file,
            h5_dir=hierarchical_h5_dir,
            split='val',
        )
        collate_fn = hierarchical_collate_fn
    else:
        predict_dataset = RepresentationsDataset(
            csv_path=train_split_file,
            representation_dir=hierarchical_h5_dir,
            max_tiles=None,
            split='val',
        )
        collate_fn = None

    dataloader = DataLoader(
        predict_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load the MIL model
    model = MILTrainer().load_from_checkpoint(args.check_path)
    
    # Create the trainer
    trainer = L.Trainer(
        accelerator='gpu',
        devices=1)

    
    predictions = trainer.predict(model, dataloader)
