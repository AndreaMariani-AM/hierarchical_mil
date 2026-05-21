# callbacks/best_metrics_checkpoint.py
from __future__ import annotations
import copy
import logging
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl

log = logging.getLogger(__name__)


def _to_float_safe(x: Any) -> Optional[float]:
    """Convert a tensor/int/float to float safely, return None if not convertible or is nan."""
    try:
        if x is None:
            return None
        if hasattr(x, "item"):
            v = float(x.item())
        elif isinstance(x, (float, int)):
            v = float(x)
        else:
            return None
        if v != v:  # NaN check
            return None
        return v
    except Exception:
        return None


class BestMetricsCheckpointCallback(pl.Callback):
    """
    Record the metrics of the epoch that becomes the new 'best' according to a ModelCheckpoint.

    Behavior:
    - On validation epoch end (after metrics are logged) we check the ModelCheckpoint callback(s)
      in the trainer for monitored metric and its mode (min/max).
    - If the current monitored metric is considered better than the stored best_model_score,
      we snapshot current aggregated metrics (trainer.callback_metrics) as best_metrics.
    - On saving a checkpoint (on_save_checkpoint) we inject the latest best_metrics into checkpoint
      under `checkpoint_key` (default: "best_metrics").
    - On loading a checkpoint (on_load_checkpoint) we restore best_metrics to the callback instance.

    Notes:
    - This callback expects to be used alongside ModelCheckpoint(s). It will try to find an existing
      ModelCheckpoint instance that monitors the same metric; if multiple exist, it uses the first matched.
    - The snapshot contains:
        {"epoch": int, "step": int, "monitor": "<monitor_tag>", "monitor_value": float, "metrics": {tag: float}}
    """

    def __init__(
        self,
        checkpoint_key: str = "best_metrics",
        monitor: Optional[str] = None,
        restrict_tags_prefix: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_key: key used to store the best-metrics snapshot inside the checkpoint dict.
            monitor: optional explicit monitor tag (e.g. "val/loss"). If None, it will read from the
                     ModelCheckpoint callback's `monitor` attribute.
            restrict_tags_prefix: if set (e.g. "val/") only tags starting with this prefix are stored.
                                  If None, all scalar tags in trainer.callback_metrics are considered.
        """
        super().__init__()
        self.checkpoint_key = checkpoint_key
        self._explicit_monitor = monitor
        self.restrict_tags_prefix = restrict_tags_prefix

        # Stored snapshot for the best epoch so far
        # None or dict as described in class docstring
        self.best_metrics: Optional[Dict[str, Any]] = None

    def _find_checkpoint_cb(self, trainer: pl.Trainer) -> Optional[pl.callbacks.model_checkpoint.ModelCheckpoint]:
        """
        Find a ModelCheckpoint callback in trainer.callbacks that monitors the requested metric.
        Preference order:
          1) If explicit monitor provided, find first checkpoint with same monitor.
          2) Else return first ModelCheckpoint in trainer.callbacks.
        """
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.model_checkpoint.ModelCheckpoint):
                if self._explicit_monitor is None:
                    return cb
                else:
                    try:
                        if getattr(cb, "monitor", None) == self._explicit_monitor:
                            return cb
                    except Exception:
                        continue
        return None

    def _is_current_better_than_best(self, current: float, best: Optional[float], mode: str) -> bool:
        """
        Compare using 'min' or 'max' semantics. None best means current is better by default.
        """
        if current is None:
            return False
        if best is None:
            return True
        if mode == "min":
            return current < best
        else:
            return current > best

    def _collect_metrics_snapshot(self, trainer: pl.Trainer, monitor_tag: str, monitor_value: float) -> Dict[str, Any]:
        # epoch/global_step info
        epoch = getattr(trainer, "current_epoch", None)
        step = getattr(trainer, "global_step", None)

        # trainer.callback_metrics contains aggregated metrics available to callbacks
        all_metrics = trainer.callback_metrics or {}

        # normalize to float scalars (skip non-scalars)
        metrics_out = {}
        for k, v in all_metrics.items():
            # filter out entries that are not scalar convertible or are dataloader references, etc.
            fv = _to_float_safe(v)
            if fv is None:
                continue
            if self.restrict_tags_prefix is not None and not k.startswith(self.restrict_tags_prefix):
                continue
            metrics_out[k] = fv

        snapshot = {
            "epoch": int(epoch) if epoch is not None else None,
            "step": int(step) if step is not None else None,
            "monitor": monitor_tag,
            "monitor_value": float(monitor_value) if monitor_value is not None else None,
            "metrics": metrics_out,
        }
        return snapshot

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """After validation epoch — check monitored metric and update best_metrics if improved."""
        ckpt_cb = self._find_checkpoint_cb(trainer)
        if ckpt_cb is None:
            # no checkpoint callback found — nothing to compare to
            return

        monitor_tag = self._explicit_monitor if self._explicit_monitor is not None else getattr(ckpt_cb, "monitor", None)
        if monitor_tag is None:
            # ModelCheckpoint has no monitor configured
            return

        # Extract current monitored value from trainer.callback_metrics
        raw_val = trainer.callback_metrics.get(monitor_tag)
        current_val = _to_float_safe(raw_val)

        # Get current best score from ModelCheckpoint; may be None
        best_score = getattr(ckpt_cb, "best_model_score", None)
        best_value = _to_float_safe(best_score)

        # Mode determination: ModelCheckpoint.mode may be "min" or "max"
        mode = getattr(ckpt_cb, "mode", "min")
        # Some versions use "min" or "max"; ensure normalized
        mode = "min" if mode == "min" else "max"

        if current_val is None:
            # cannot compare
            return

        is_better = self._is_current_better_than_best(current_val, best_value, mode)

        # If ckpt_cb monitors step-based saving (like save_top_k), ModelCheckpoint will update its internals
        # later. We just record that this epoch produced a better monitored value.
        if is_better:
            # snapshot everything we want to save for this epoch
            self.best_metrics = self._collect_metrics_snapshot(trainer, monitor_tag, current_val)
            log.debug(f"New best metrics recorded for monitor {monitor_tag}: {self.best_metrics}")

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Inject best_metrics snapshot into the checkpoint if present.
        Called when ANY checkpoint is saved (ModelCheckpoint or manual trainer.save_checkpoint).
        We only inject the single best-metrics snapshot (if available).
        """
        if self.best_metrics is not None:
            # Make a defensive deep copy to avoid references to tensors, etc.
            checkpoint = checkpoint or {}
            checkpoint[self.checkpoint_key] = copy.deepcopy(self.best_metrics)
        return checkpoint

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]) -> None:
        saved = checkpoint.get(self.checkpoint_key)
        if saved is not None:
            self.best_metrics = saved