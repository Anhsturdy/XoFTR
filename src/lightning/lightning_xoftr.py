from collections import defaultdict
import pprint
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from src.xoftr import XoFTR
from src.xoftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.xoftr_loss import XoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_XoFTR(pl.LightningModule):
    """Lightning wrapper for XoFTR compatible with PyTorch‑Lightning ≥ 2.0."""

    # ------------------------------------------------------------------
    # INIT --------------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, config, pretrained_ckpt: str | None = None,
                 profiler=None, dump_dir: str | None = None):
        super().__init__()

        # 1) store config -------------------------------------------------
        self.config = config
        _cfg = lower_config(config)
        self.xoftr_cfg = lower_config(_cfg["xoftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # 2) build model --------------------------------------------------
        self.matcher = XoFTR(config=_cfg["xoftr"])
        self.loss_fn = XoFTRLoss(_cfg)

        if pretrained_ckpt:
            state = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            miss, unexp = self.matcher.load_state_dict(state, strict=False)
            logger.info(f"Loaded checkpoint {pretrained_ckpt}")
            if miss:
                logger.warning(f"Missing keys: {miss}")
            if unexp:
                logger.warning(f"Unexpected keys: {unexp}")

        # 3) misc ---------------------------------------------------------
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.validation_step_outputs: list = []
        self.test_step_outputs: list = []
        self.train_losses: list = []

    # ------------------------------------------------------------------
    # OPTIMISERS --------------------------------------------------------
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optim = build_optimizer(self, self.config)
        sched = build_scheduler(self.config, optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step" if self.config.TRAINER.LR_SCHED_BY_STEP else "epoch",
            },
        }

    # custom LR warm‑up --------------------------------------------------
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        warmup_steps = self.config.TRAINER.WARMUP_STEP
        if self.global_step < warmup_steps and self.config.TRAINER.WARMUP_TYPE == "linear":
            base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
            lr = base_lr + (self.global_step / warmup_steps) * (
                self.config.TRAINER.TRUE_LR - base_lr
            )
            for g in optimizer.param_groups:
                g["lr"] = lr
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    # ------------------------------------------------------------------
    # INTERNAL UTILS ----------------------------------------------------
    # ------------------------------------------------------------------
    def _trainval_forward(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
        with self.profiler.profile("Compute losses"):
            self.loss_fn(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)
            rel_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                "identifiers": ["#".join(rel_names[b]) for b in range(bs)],
                "epi_errs"  : [batch["epi_errs"][batch["m_bids"] == b].cpu().numpy() for b in range(bs)],
                "R_errs"    : batch["R_errs"],
                "t_errs"    : batch["t_errs"],
                "inliers"   : batch["inliers"],
            }
            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics["scene_id"] = batch["scene_id"]
            return {"metrics": metrics}, rel_names

    # ------------------------------------------------------------------
    # TRAINING ----------------------------------------------------------
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        self._trainval_forward(batch)
        self.train_losses.append(batch["loss"].detach())

        if self.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in batch["loss_scalars"].items():
                self.log(f"train/{k}", v, on_step=True, prog_bar=False)
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)
                figs = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, fig in figs.items():
                    self.logger.experiment.add_figure(f"train_match/{k}", fig, self.global_step)
        return batch["loss"]

    def on_train_epoch_end(self):
        if not self.train_losses:
            return
        avg = torch.stack(self.train_losses).mean()
        self.log("train/avg_loss_on_epoch", avg, prog_bar=True)
        self.train_losses.clear()

    # ------------------------------------------------------------------
    # VALIDATION --------------------------------------------------------
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            with self.profiler.profile("XoFTR"):
                self.matcher(batch)
        else:
            self._trainval_forward(batch)

        ret, _ = self._compute_metrics(batch)
        if self.config.DATASET.VAL_DATA_SOURCE != "VisTir":
            ret["loss_scalars"] = batch["loss_scalars"]

        interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        ret["figures"] = make_matching_figures(
            batch, self.config, self.config.TRAINER.PLOT_MODE, ret
        ) if batch_idx % interval == 0 else {self.config.TRAINER.PLOT_MODE: []}

        self.validation_step_outputs.append(ret)
        return ret

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        outs = self.validation_step_outputs
        multi_outputs = [outs] if not isinstance(outs[0], (list, tuple)) else outs
        multi_val_metrics = defaultdict(list)

        for valset_idx, out in enumerate(multi_outputs):
            cur_epoch = -1 if self.trainer.sanity_checking else self.current_epoch

            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics_per_scene = defaultdict(list)
                for o in out:
                    metrics_per_scene[o["metrics"]["scene_id"][0]].append(o["metrics"])

                aucs_per_scene = {}
                for scene_id, scene_metrics in metrics_per_scene.items():
                    metrics = {
                        k: flattenList(all_gather(flattenList([m[k] for m in scene_metrics])))
                        for k in scene_metrics[0]
                    }
                    aucs_per_scene[scene_id] = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

                val_metrics_tb = {}
                for thr in [5, 10, 20]:
                    val_metrics_tb[f"auc@{thr}"] = float(np.mean([aucs_per_scene[s][f"auc@{thr}"] for s in aucs_per_scene]))
                val_metrics_tb[f"prec@{self.config.TRAINER.EPI_ERR_THR:.0e}"] = float(np.mean([
                    aucs_per_scene[s][f"prec@{self.config.TRAINER.EPI_ERR_THR:.0e}"] for s in aucs_per_scene]))
                loss_scalars = {}
            else:
                loss_scalars = {
                    k: flattenList(all_gather(flattenList([o["loss_scalars"][k] for o in out])))
                    for k in out[0]["loss_scalars"]
                }
                metrics = {
                    k: flattenList(all_gather(flattenList([o["metrics"][k] for o in out])))
                    for k in out[0]["metrics"]
                }
                val_metrics_tb
