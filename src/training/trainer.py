# Main lighting trainer class for MIL model
import os
from pathlib import Path
import sys

from typing import Dict, Any
sys.path.append(os.path.abspath('/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src')) 
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
# from src.models.MIL import AttentionMIL
from models.MIL import AttentionMIL, AdditiveMIL, HierarchicalMIL
from src.training.losses import FocalLoss, FocalTverskyLoss

VIRCHOW2DIM = 1280
UNI2DIM = 1536


class MILTrainer(L.LightningModule):
    def __init__(
            self,
            input_dim: int = 2560,
            hidden_dim: int = 1280,
            hidden_dim_2: int = 256,
            n_classes: int = 1,
            dropout: float = 0.25,
            lr: float = 1e-4,
            weight_decay: float = 1e-5,
            instance_batch_size: int = 512,
            model_type: str = "attention",
            pooling: str = "sum",
            frozen_backbone: str = 'Virchow2',
            top_k: int = 512,
            scale_drop_p: float = 0.15, #all 3 61% of steps, 1 dropped 33% of steps an 2 dropped 6% of steps
            cell_warmup_start: int = 20, # activate CellExpert
            cell_warmup_epochs: int = 10,   # epochs at reduced lr before full training
            cell_warmup_lr_factor: float = 0.1,
            loss: str = "bce",
            patient_level: bool = False,
    ):
        """
        Lightning wrapper for MIL models.
        
        :param input_dim: Dimension of input instance features
        :type input_dim: int
        :param hidden_dim: First hidden layer dimension
        :type hidden_dim: int
        :param hidden_dim_2: Second hidden layer dimension
        :type hidden_dim_2: int
        :param n_classes: Number of classes. 1 for binary (BCE), >1 for multi-class (CE)
        :type n_classes: int
        :param dropout: Dropout rate
        :type dropout: float
        :param lr: Learning rate
        :type lr: float
        :param instance_batch_size: Batch size for instance processing
        :type instance_batch_size: int
        :param model_type: Model architecture - "attention" or "additive" or "hierarchical"
        :type model_type: str
        :param pooling: Pooling strategy for AdditiveMIL - "sum", "mean", or "max"
        :type pooling: str
        :param frozen_backbone: Name of the frozen backbone to use to extract patch tokens - 'Virchow2' or 'UNI2'
        :type frozen_backbone: str
        :param top_k: Number of top patches to select for CellExpert
        :type top_k: int
        :param scale_drop_p: Probability of dropping a scale in hierarchical model
        :type scale_drop_p: float
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self._is_hierarchical = False
        self._is_patient_level = patient_level
        self.cell_warmup_start = cell_warmup_start
        self.cell_warmup_epochs = cell_warmup_epochs
        self.cell_warmup_lr_factor = cell_warmup_lr_factor
        self._cell_expert_active = False
        self.validation_step_outputs: list = []  # accumulates per-sample {slide_id, loss} for loss distribution analysis

        # Conditionally instantiate model
        if model_type == "attention":
            self.model = AttentionMIL(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_dim_2=hidden_dim_2,
                n_classes=n_classes,
                dropout=dropout,
                instance_batch_size=instance_batch_size
            )
        elif model_type == "additive":
            self.model = AdditiveMIL(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_dim_2=hidden_dim_2,
                n_classes=n_classes,
                dropout=dropout,
                instance_batch_size=instance_batch_size,
                pooling=pooling
            )
        elif model_type == "hierarchical":
            if frozen_backbone == 'Virchow2':
                cell_token_dim = VIRCHOW2DIM
            else:
                cell_token_dim = UNI2DIM

            self.model = HierarchicalMIL(
                region_in_dim=input_dim,      # 2560 for Virchow2 (CLS+mean)
                patch_in_dim=cell_token_dim,  # 1280/1536 for Virchow2/UNI2 (CLS token)
                cell_token_dim=cell_token_dim,
                hidden_dim_2=hidden_dim_2,
                n_classes=n_classes,
                dropout=dropout,
                instance_batch_size=instance_batch_size,
                pooling=pooling,
                top_k=top_k,
                scale_drop_p=scale_drop_p,
            )
            self._is_hierarchical = True
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'attention', 'additive', or 'hierarchical'.")

        # Conditional loss: BCE for binary (n_classes=1), CrossEntropy for multi-class
        if loss == "bce":
            if n_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        if loss == 'tversky':
            self.criterion = FocalTverskyLoss()
        
        if loss == 'focal':
            self.criterion = FocalLoss()

        self.lr = lr
        self.weight_decay = weight_decay
        
        # Conditional metrics based on n_classes
        if n_classes == 1:
            task_kwargs = {"task": "binary"}
        else:
            task_kwargs = {"task": "multiclass", "num_classes": n_classes}

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(**task_kwargs),
            }, prefix="train_"
            )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(**task_kwargs),
                "F1": torchmetrics.classification.F1Score(**task_kwargs),
                "AUROC": torchmetrics.classification.AUROC(**task_kwargs),
            }, prefix="val_"
            )


    @property
    def cell_expert_active(self) -> bool:
        """
        Becomes True when current epoch is > than cell_warmup_start
        """
        return self.current_epoch >= self.cell_warmup_start

    def _aux_loss_MoE(
        self,
        attn_dict: Dict[str, Any],
        label: torch.Tensor,
        n_instances: Dict[str, int],
    ):
        """
        Auxiliary loss to encourage all experts to learn meaningful representations.
        For each expert, we compute a loss between its individual prediction and the true label.
        Logits are normalised by sqrt(N_instances) for consistency with the fusion head.
        """
        aux_loss = 0.0
        experts = ['region', 'patch']
        if self.cell_expert_active:
            experts.append('cell')

        for expert_name in experts:
            n = n_instances[expert_name]
            expert_logits = attn_dict[expert_name].logits / (n ** 0.5 + 1e-8)
            if self.n_classes == 1:
                expert_loss = self.criterion(expert_logits, label.float())
            else:
                expert_loss = self.criterion(expert_logits.unsqueeze(0), label.long())
            aux_loss += expert_loss
            self.log(f'Train_{expert_name}_aux_loss', expert_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=False)

        return aux_loss
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep h_cells on CPU to avoid OOM; transferred lazily in HierarchicalMIL.forward()."""
        # Patient-level: slides_list stays on CPU; only label is moved.
        # Must short-circuit before super() which would move all slide tensors to GPU.
        if self._is_patient_level and isinstance(batch, (tuple, list)) and len(batch) == 3:
            slides_list, label, patient_id = batch
            label = label.to(device) if isinstance(label, torch.Tensor) else label
            return (slides_list, label, patient_id)
        if self._is_hierarchical and isinstance(batch, (tuple, list)) and len(batch) == 7:
            h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions = batch
            h_patches  = h_patches.to(device)  if isinstance(h_patches,  torch.Tensor) else h_patches
            h_region   = h_region.to(device)   if isinstance(h_region,   torch.Tensor) else h_region
            # h_cells intentionally stays on CPU — indexed and moved lazily in HierarchicalMIL.forward()
            label      = label.to(device)      if isinstance(label,      torch.Tensor) else label
            region_ids = region_ids.to(device) if isinstance(region_ids, torch.Tensor) else region_ids
            return (h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _process_patient_batch(
        self,
        slides_list,
        activate_cell_expert: bool,
    ):
        """
        Two-pass patient-level forward for ``max(|logit|)`` winner selection.

        Pass 1 (no_grad): run all slides to find the most extreme logit.
        Pass 2 (grad):    re-run only the winning slide with attn_return=True.

        Parameters
        ----------
        slides_list : list of (h_patches, h_region, h_cells, region_ids, n_regions, fetch_fn)
            Raw CPU tensors per slide from ``PatientHierarchicalDataset.__getitem__``.
        activate_cell_expert : bool
            Whether to pass cell tokens to CellExpert in this call.

        Returns
        -------
        winner_logit : Tensor ``(1,)``
        attn_dict    : dict
        winner_idx   : int
        """
        device = self.device

        # ── Pass 1: no_grad scan over all slides ─────────────────────
        per_slide_logits = []
        with torch.no_grad():
            for (h_patches, h_region, h_cells, region_ids, n_regions) in slides_list:
                h_patches_d  = h_patches.unsqueeze(0).to(device)   # (1, N, D_p)
                h_region_d   = h_region.unsqueeze(0).to(device)    # (1, N, D_r)
                region_ids_d = region_ids.to(device)               # (N,) — no unsqueeze
                h_cell_tokens = (
                    h_cells.unsqueeze(0).to(device) if activate_cell_expert else None
                )
                logit, _ = self(
                    h_patches_d, attn_return=False,
                    region_ids=region_ids_d,
                    n_regions=n_regions,
                    h_region_features=h_region_d,
                    h_cell_tokens=h_cell_tokens,
                )
                per_slide_logits.append(logit.detach().cpu())  # (1,) binary

        logits_stacked = torch.stack(per_slide_logits).squeeze(-1)  # (N_slides,)
        winner_idx = int(logits_stacked.abs().argmax().item())
        # max(|logit|): label-agnostic; same rule at train and inference;
        # picks the most-extreme slide regardless of class direction.

        # ── Pass 2: grad-enabled forward on winner ───────────────────
        h_patches, h_region, h_cells, region_ids, n_regions = slides_list[winner_idx]
        h_patches_d  = h_patches.unsqueeze(0).to(device)
        h_region_d   = h_region.unsqueeze(0).to(device)
        region_ids_d = region_ids.to(device)
        h_cell_tokens = (
            h_cells.unsqueeze(0).to(device) if activate_cell_expert else None
        )
        winner_logit, attn_dict = self(
            h_patches_d, attn_return=True,
            region_ids=region_ids_d,
            n_regions=n_regions,
            h_region_features=h_region_d,
            h_cell_tokens=h_cell_tokens,
        )

        return winner_logit, attn_dict, winner_idx

    def forward(
            self,
            x: torch.Tensor,
            attn_return: bool = False,
            **kwargs,
    ):
        """Dispatch to the underlying MIL model.

        For hierarchical models, extra kwargs (region_ids, n_regions,
        h_cell_tokens) are forwarded to ``HierarchicalMIL.forward``.
        """
        if self._is_hierarchical:
            return self.model(
                h_patches=x,
                region_ids=kwargs.get("region_ids"),
                n_regions=kwargs.get("n_regions"),
                h_region_features=kwargs.get("h_region_features"),
                h_cell_tokens=kwargs.get("h_cell_tokens"),
                attn_return=attn_return,
                return_region_emb=kwargs.get("return_region_emb", False)
            )
        return self.model(x, attn_return)
    
    def training_step(
            self,
            batch,
            batch_idx
    ):
        
        # ── Unpack batch ─────────────────────────────────────────
        # Patient-level:    (slides_list, label, patient_id)
        # Hierarchical:     (h_patches, h_region, h_cells, label, slide_id,
        #                    region_ids, n_regions)
        # Non-hierarchical: (features, label, slide_id)
        if self._is_patient_level:
            slides_list, label, patient_id = batch
            logits, attn_dict, _ = self._process_patient_batch(
                slides_list, activate_cell_expert=self.cell_expert_active
            )
        elif self._is_hierarchical:
            h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions = batch

            # Apply curriculum: only activate CellExpert once warmup starts
            h_cell_tokens_active = h_cells if self.cell_expert_active else None

            logits, attn_dict = self(
                h_patches, attn_return=True,
                region_ids=region_ids,
                n_regions=n_regions,
                h_region_features=h_region,
                h_cell_tokens=h_cell_tokens_active,
            )
        else:
            features, label, slide_id = batch
            logits, attn_dict, _ = self(features, attn_return=True)
        

        # Conditional label casting: float for BCE, long for CrossEntropy
        if self.n_classes == 1:
            loss = self.criterion(logits, label.float())
        else:
            loss = self.criterion(logits.unsqueeze(0), label.long())
        
        if self._is_hierarchical:
            n_instances = attn_dict['n_instances']
            aux_loss = self._aux_loss_MoE(attn_dict, label, n_instances)
            loss = loss + aux_loss
            self.log('train_aux_loss', aux_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=False)

        train_metrics = self.train_metrics(logits, label.long())
        
        self.log('train_loss_main', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=True)
        self.log_dict(train_metrics, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=True)
        self.log('train_logit_abs_max', logits.detach().abs().max(), on_step=True, sync_dist=False, batch_size=1)
        return loss
    
    def on_train_epoch_start(self):
        """
        Adjust the CellExpert lr according to the curriculum
        """
        if self._is_hierarchical:
            optimizer = self.optimizers()
            for pg in optimizer.param_groups:
                if pg.get("name") == "cell_expert":
                    epoch = self.current_epoch

                    if epoch < self.cell_warmup_start:
                        target_lr = 0.0 # inactive

                    elif epoch < self.cell_warmup_start + self.cell_warmup_epochs:
                        # warmup at reduced lr
                        target_lr = self.lr * self.cell_warmup_lr_factor

                    else:
                        target_lr = self.lr

                    # save the new LR
                    if pg["lr"] != target_lr:
                        pg["lr"] = target_lr
                        self.log("cell_expert_lr", target_lr, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=False)
                    break
        
        # Log status of CellExpert (0=inactive, 1=warming up, 2=full)
        epoch = self.current_epoch
        if epoch < self.cell_warmup_start:
            phase = 0
        elif epoch < self.cell_warmup_start + self.cell_warmup_epochs:
            phase = 1
        else:
            phase = 2
        
        self.log("CellExpert_phase", float(phase), on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=False)
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def validation_step(
            self,
            batch,
            batch_idx
    ):
        # ── Unpack batch (same logic as training_step) ───────────
        if self._is_patient_level:
            slides_list, label, sample_id = batch
            activate = self.current_epoch >= (self.cell_warmup_start + self.cell_warmup_epochs)
            logits, _, _ = self._process_patient_batch(
                slides_list, activate_cell_expert=activate
            )
        elif self._is_hierarchical:
            h_patches, h_region, h_cells, label, sample_id, region_ids, n_regions = batch

            # CellExpert active only after warmup is complete
            h_cell_tokens_active = h_cells if (
                self.current_epoch >= (self.cell_warmup_start + self.cell_warmup_epochs)
            ) else None

            logits, _ = self(
                h_patches, attn_return=False,
                region_ids=region_ids,
                n_regions=n_regions,
                h_region_features=h_region,
                h_cell_tokens=h_cell_tokens_active,
            )
        else:
            features, label, sample_id = batch
            logits, _ , _= self(features, attn_return=False)
        
        # Conditional label casting (same smoothing as training for comparable losses)
        if self.n_classes == 1:
            loss_val = self.criterion(logits, label.float())
        else:
            loss_val = self.criterion(logits.unsqueeze(0), label.long())

        self.val_metrics.update(logits, label.long())
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=True)

        # After computing logits in validation_step
        self.log('val_logit_mean', logits.mean(), on_step=True, sync_dist=False, batch_size=1)
        self.log('val_logit_abs_max', logits.abs().max(), on_step=True, sync_dist=False, batch_size=1)
        self.validation_step_outputs.append({"slide_id": sample_id, "loss": loss_val.detach().cpu().item()})
        return loss_val
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1, prog_bar=True)
        self.val_metrics.reset()

        # Store as instance variables for on_save_checkpoint
        # Read directly from computed dict — no dependency on callback_metrics timing
        self.current_val_loss     = self.trainer.callback_metrics['val_loss'].item()
        self.current_val_accuracy = self.trainer.callback_metrics['val_accuracy'].item()
        self.current_val_F1       = self.trainer.callback_metrics['val_F1'].item()
        self.current_val_AUROC    = self.trainer.callback_metrics['val_AUROC'].item()
    
    @staticmethod
    def _detach_attn_val(v):
        """Detach and move to CPU, handling ExpertOutput dataclasses and plain tensors."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if hasattr(v, "__dataclass_fields__"):
            return {
                field: getattr(v, field).detach().cpu()
                if isinstance(getattr(v, field), torch.Tensor)
                else getattr(v, field)
                for field in v.__dataclass_fields__
            }
        return v

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Similar unpacking logic as training/validation steps
        if self._is_hierarchical:
            h_patches, h_region, h_cells, label, slide_id, region_ids, n_regions = batch

            h_cell_tokens_active = h_cells if self.cell_expert_active else None

            logits, attn_dict = self(
                h_patches, attn_return=True,
                region_ids=region_ids,
                n_regions=n_regions,
                h_region_features=h_region,
                h_cell_tokens=h_cell_tokens_active,
                return_region_emb=True
            )

        else:
            features, label, slide_id = batch
            logits, attn_weights, contributions = self(features, attn_return=True)
            attn_dict = {"attn_weights": attn_weights, "contributions": contributions}

        return {
            "logits": logits.detach().cpu(),
            "attn_dict": {k: self._detach_attn_val(v) for k, v in attn_dict.items()},
            "label": label.detach().cpu(),
            "slide_id": slide_id
        }
    
    def configure_optimizers(self):
        # Only optimise trainable parameters — frozen backbone weights
        # (if any were ever registered) are excluded via requires_grad.
        # LR of CellExpert is scaled down to 0 untill it can be trained
        # i need to split the params
        if self._is_hierarchical:
            cell_params = list(self.model.cell_expert.parameters())
            cell_params_ids = {id(p) for p in cell_params}

            other_params = [
                p for p in self.parameters() if p.requires_grad and id(p) not in cell_params_ids
                ]
            cell_trainable_params = [p for p in cell_params if p.requires_grad]

            # CellExpert starts with 0 LR and is updated in on_train_epoch_start
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params,  "lr": self.lr},
                    {"params": cell_trainable_params, "lr": 0.0, "name": "cell_expert"},
                ], 
                weight_decay=self.weight_decay
            )
        else:
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        # Save the latest epoch metrics into the checkpoint
        checkpoint['val_loss']     = self.current_val_loss
        checkpoint['val_accuracy'] = self.current_val_accuracy
        checkpoint['val_F1']       = self.current_val_F1
        checkpoint['val_AUROC']    = self.current_val_AUROC