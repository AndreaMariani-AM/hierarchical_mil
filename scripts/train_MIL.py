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
    PatientHierarchicalDataset,
    patient_hierarchical_collate_fn,
)
from torch.utils.data import DataLoader
import torch
import lightning as L
from src.training.trainer import MILTrainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.strategies import DDPStrategy

torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)  # if using multi-GPU

# model definition
# training loop

if __name__ == '__main__':
    """
    Main training script.
    """
    t = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Path to config file')
    parser.add_argument('--fold', type=str, default=None, help='Fold number')
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['attention', 'additive', 'hierarchical'],
                        help='MIL model type: "attention" or "additive" or "hierarchical" (default: attention)')
    parser.add_argument('--pooling', type=str, default='sum',
                        choices=['sum', 'mean', 'max'],
                        help='Pooling strategy for AdditiveMIL (default: sum)')
    parser.add_argument('--frozen_backbone', type=str, default='Virchow2',
                        choices=['Virchow2', 'UNI2'],
                        help='Frozen backbone to extract dense patch tokens')
    parser.add_argument('--top_k', type=int, default=128, 
                        help='Top k attended patches to use for Cell Expert in HierarchicalMIL (default: 128)')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'tversky', 'focal'],
                        help='Loss function to use: "bce" for binary, "tversky" for Tversky loss, "focal" for Focal loss (default: bce)')
    parser.add_argument('--scale_drop_p', type=float, default=0.15, 
                        help='Probability of dropping a scale in HierarchicalMIL (default: 0.15)')
    parser.add_argument('--max_epochs', type=int, default=100, 
                        help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=32, 
                        help='Number of batches to accumulate gradients over (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for the optimizer (default: 1e-3)')
    parser.add_argument('--input_dim', type=int, default=2560,
                        help='Input dimension size for the MIL model (default: 2560)')
    parser.add_argument('--hidden_dim', type=int, default=1280,
                        help='Hidden dimension size for the MIL model (default: 1280)')
    parser.add_argument('--hidden_dim_2', type=int, default=256,
                        help='Common final dimension for all MIL Experts (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate for the MIL model (default: 0.25)')
    parser.add_argument('--cell_warmup_start', type=int, default=20,
                        help='Epoch to activate CellExpert (default: 20)')
    parser.add_argument('--cell_warmup_epochs', type=int, default=10,
                        help='Epoch to warmup CellExpert (default: 10)')
    parser.add_argument('--cell_warmup_lr_factor', type=int, default=0.1,
                        help='LR multiplier for CellExpert during warm-up (default: 0.1)')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment to add to the experiment name (default: empty)')
    parser.add_argument('--patient_level', action='store_true',
                        help='Index over patients; winner slide selected by max(|logit|) (default: True)')
    
    args = parser.parse_args()

    # read config file
    with open(args.config_file, "r") as file_yml:
        config_file = yaml.safe_load(file_yml)

    folds_dir = Path(config_file['training']['folds_dir'])
    hierarchical_h5_dir = Path(config_file['training']['hierarchical_h5_dir'])
    outdir = Path(config_file['training']['out_dir'])
    
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    
    fold_num=args.fold
    # fold_num=0# For now, just use fold 0; can be parameterized later
    train_split_file = folds_dir / f'fold_{fold_num}.csv'

    # create dataset and dataloader
    if args.model_type == 'hierarchical':
        if args.patient_level:
            train_dataset = PatientHierarchicalDataset(
                csv_path=train_split_file,
                h5_dir=hierarchical_h5_dir,
                split='train',
            )
            val_dataset = PatientHierarchicalDataset(
                csv_path=train_split_file,
                h5_dir=hierarchical_h5_dir,
                split='val',
            )
            collate_fn = patient_hierarchical_collate_fn
        else:
            train_dataset = HierarchicalRepresentationsDataset(
                csv_path=train_split_file,
                h5_dir=hierarchical_h5_dir,
                split='train',
            )
            val_dataset = HierarchicalRepresentationsDataset(
                csv_path=train_split_file,
                h5_dir=hierarchical_h5_dir,
                split='val',
            )
            # Always use the custom collate — it handles callable fetch_fn=None
            # gracefully and is safer than default collate for this batch structure.
            collate_fn = hierarchical_collate_fn
    
    else:
        train_dataset = RepresentationsDataset(
            csv_path=train_split_file,
            representation_dir=hierarchical_h5_dir,
            max_tiles=None,
            split='train',
        )
        val_dataset = RepresentationsDataset(
            csv_path=train_split_file,
            representation_dir=hierarchical_h5_dir,
            max_tiles=None,
            split='val',
        )
        collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Load the MIL model
    model = MILTrainer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        hidden_dim_2=args.hidden_dim_2,
        n_classes=1,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        instance_batch_size=1024,
        model_type=args.model_type,
        pooling=args.pooling,
        frozen_backbone=args.frozen_backbone,
        top_k=args.top_k,
        scale_drop_p=args.scale_drop_p,
        cell_warmup_start=args.cell_warmup_start,
        cell_warmup_epochs=args.cell_warmup_epochs,
        cell_warmup_lr_factor=args.cell_warmup_lr_factor,
        loss=args.loss,
        patient_level=args.patient_level,
    )
    
# Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=outdir,
        monitor="val_AUROC",
        save_top_k=1,
        mode="max",
        filename=f"fold_{fold_num}" + "-{epoch:02d}-{val_AUROC:.2f}" + f"-{args.model_type}_{args.comment}")
    
    early_stopping_callback = EarlyStopping(
        monitor="val_AUROC",
        patience=100, 
        mode="max",
        min_delta=0.002
    )

    # logger = TensorBoardLogger(outdir / "tb_logs", name=f"fold_{fold_num}_{args.model_type}_{args.comment}")
    logger = WandbLogger(project="IBD_Hierarchical_MIL", name=f"fold_{fold_num}_{args.model_type}_{args.comment}", 
                         save_dir=outdir, log_model=True)

    # Create the trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator='gpu',
        devices=1,
        # num_nodes=2,
        # strategy=DDPStrategy(find_unused_parameters=True),
        precision='16-mixed',  #can only use 16-mixed precision ov V100 GPUs and not bf16 precision
        detect_anomaly=False)  # set True temporarily if you see nan losses)
    
    #limit_train_batches=0.1, limit_val_batches=0.01

    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    # Check best model's metrics
    best_path = checkpoint_callback.best_model_path
    checkpoint  = torch.load(best_path, map_location='cpu')

    metrics = {
        "Validation Loss":     checkpoint['val_loss'],
        "Validation Accuracy": checkpoint['val_accuracy'],
        "Validation F1":       checkpoint['val_F1'],
        "Validation AUROC":    checkpoint['val_AUROC'],
    }

    col_w = 22
    print("┌" + "─" * col_w + "┬" + "─" * 12 + "┐")
    print(f"│ {'Metric':<{col_w - 2}} │ {'Value':<10} │")
    print("├" + "─" * col_w + "┼" + "─" * 12 + "┤")
    for name, val in metrics.items():
        print(f"│ {name:<{col_w - 2}} │ {val:<10.4f} │")
    print("└" + "─" * col_w + "┴" + "─" * 12 + "┘")