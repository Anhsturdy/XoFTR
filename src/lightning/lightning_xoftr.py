from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from src.xoftr import XoFTR
from src.xoftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.xoftr_loss import XoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_XoFTR(pl.LightningModule):
    """LightningModule wrapper around XoFTR that guarantees the AUC metrics are
    always logged so that ModelCheckpoint(monitor="auc@10") never fails.
    """

    def __init__(self, config, pretrained_ckpt: str | None = None, profiler=None, dump_dir: str | None = None):
        super().__init__()

        # --------------------------------------------------
        # Misc / bookkeeping
        # --------------------------------------------------
        self.save_hyperparameters({"cfg": config})  # saves "cfg" in the checkpoint for reproducibility
        self.config = config
        _config = lower_config(config)
        self.xoftr_cfg = lower_config(_config["xoftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Will accumulate per‑batch losses so that we can log an epoch‑level value.
        self.train_losses: list[torch.Tensor] = []
        self.latest_val_auc: dict[int, float] = {5: 0.0, 10: 0.0, 20: 0.0}  # safety store

        # --------------------------------------------------
        # Core model + loss
        # --------------------------------------------------
        self.matcher = XoFTR(config=_config["xoftr"])
        self.loss = XoFTRLoss(_config)

        # --------------------------------------------------
        # (Optional) load pretrained weights
        # --------------------------------------------------
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {pretrained_ckpt}")

        # --------------------------------------------------
        # Inference‑time dump dir
        # --------------------------------------------------
        self.dump_dir = dump_dir

    # ==================================================
    # Optimizer / LR schedule
    # ==================================================
    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """Implements learning‑rate warm‑up inside Lightning's optimizer_step hook."""
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / warmup_step
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE != "constant":
                raise ValueError(f"Unknown lr warm‑up strategy: {self.config.TRAINER.WARMUP_TYPE}")

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    # ==================================================
    # Internal helpers
    # ==================================================
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [batch["epi_errs"][batch["m_bids"] == b].cpu().numpy() for b in range(bs)],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics.update({"scene_id": batch["scene_id"]})
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    # ==================================================
    # Training loop
    # ==================================================
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # Store for epoch‑level average
        self.train_losses.append(batch["loss"].detach())

        # Step‑level logging (rank‑zero only)
        if self.trainer.is_global_zero and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in batch["loss_scalars"].items():
                self.log(f"train/{k}", v, prog_bar=False, logger=True, on_step=True, on_epoch=False)

            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)
                figs = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, fig in figs.items():
                    self.logger.experiment.add_figure(f"train_match/{k}", fig, self.global_step)

        return {"loss": batch["loss"]}

    def on_train_epoch_end(self):
        if not self.train_losses:
            return  # nothing recorded (should never happen)
        avg_loss = torch.stack(self.train_losses).mean()
        self.log("train/avg_loss_on_epoch", avg_loss, prog_bar=False, on_epoch=True)
        self.train_losses.clear()

    # ==================================================
    # Validation loop
    # ==================================================
    def validation_step(self, batch, batch_idx):
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            with self.profiler.profile("XoFTR"):
                self.matcher(batch)
        else:
            self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        # Lite figure plotting (only every n batches)
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figs = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figs = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE, ret_dict=ret_dict)

        payload = {**ret_dict, "figures": figs}
        if self.config.DATASET.VAL_DATA_SOURCE != "VisTir":
            payload["loss_scalars"] = batch["loss_scalars"]
        return payload

    def on_validation_epoch_end(self, outputs=None):
        if outputs is None:
            return

        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            cur_epoch = self.trainer.current_epoch  # log even during sanity check

            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics_per_scene: dict[int, list] = {}
                for o in outputs:
                    scene_id = o["metrics"]["scene_id"][0]
                    metrics_per_scene.setdefault(scene_id, []).append(o["metrics"])

                aucs_per_scene = {}
                for scene_id, scene_metrics in metrics_per_scene.items():
                    m = {k: flattenList(all_gather(flattenList([_m[k] for _m in scene_metrics]))) for k in scene_metrics[0]}
                    aucs_per_scene[scene_id] = aggregate_metrics(m, self.config.TRAINER.EPI_ERR_THR)

                val_metrics_4tb = {}
                for thr in (5, 10, 20):
                    val_metrics_4tb[f"auc@{thr}"] = float(np.mean([aucs_per_scene[s][f"auc@{thr}"] for s in aucs_per_scene]))
                val_metrics_4tb[f"prec@{self.config.TRAINER.EPI_ERR_THR:.0e}"] = float(
                    np.mean([aucs_per_scene[s][f"prec@{self.config.TRAINER.EPI_ERR_THR:.0e}"] for s in aucs_per_scene])
                )
            else:
                # Merge loss scalars across workers then across batches
                _loss_scalars = [o["loss_scalars"] for o in outputs]
                loss_scalars = {k: flattenList(all_gather([ls[k] for ls in _loss_scalars])) for k in _loss_scalars[0]}

                _metrics = [o["metrics"] for o in outputs]
                metrics = {k: flattenList(all_gather(flattenList([m[k] for m in _metrics]))) for k in _metrics[0]}
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

                # Log losses on rank‑zero
                if self.trainer.is_global_zero:
                    for k, v in loss_scalars.items():
                        self.log(f"val_{valset_idx}/avg_{k}", torch.stack(v).mean(), prog_bar=False, on_epoch=True)

            # Collect metrics for multi‑val averaging
            for thr in (5, 10, 20):
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # Extra logging for figures (rank‑zero only)
            if self.trainer.is_global_zero:
                figs = {k: flattenList(gather(flattenList([o["figures"][k] for o in outputs]))) for k in outputs[0]["figures"]}
                for k, vs in figs.items():
                    for idx, fig in enumerate(vs):
                        self.logger.experiment.add_figure(f"val_match_{valset_idx}/{k}/pair-{idx}", fig, cur_epoch, close=True)
                plt.close("all")

        # ---------- FINAL metric logging (this is crucial for ModelCheckpoint) ----------
        for thr in (5, 10, 20):
            mean_auc = float(np.mean(multi_val_metrics[f"auc@{thr}"]))
            self.latest_val_auc[thr] = mean_auc  # store for safety
            self.log(f"auc@{thr}", torch.tensor(mean_auc), prog_bar=True, on_epoch=True, on_step=False)

    # ==================================================
    # Testing loop
    # ==================================================
    def test_step(self, batch, batch_idx):
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
        ret_dict, rel_pair_names = self._compute_metrics(batch)

        if self.dump_dir is not None:
            with self.profiler.profile("dump_results"):
                Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf_f", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    mask = batch["m_bids"] == b_id
                    item = {
                        "pair_names": pair_names[b_id],
                        "identifier": "#".join(rel_pair_names[b_id]),
                    }
                    if self.config.DATASET.TEST_DATA_SOURCE == "VisTir":
                        item.update(
                            {
                                "scene_id": batch["scene_id"],
                                "K0": batch["K0"][b_id].cpu().numpy(),
                                "K1": batch["K1"][b_id].cpu().numpy(),
                                "dist0": batch["dist0"][b_id].cpu().numpy(),
                                "dist1": batch["dist1"][b_id].cpu().numpy(),
                            }
                        )
                    for k in keys_to_save:
                        item[k] = batch[k][mask].cpu().numpy()
                    for k in ("R_errs", "t_errs", "inliers"):
                        item[k] = batch[k][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps
        return ret_dict

    def on_test_epoch_end(self, outputs=None):
        if outputs is None:
            return

        if self.config.DATASET.TEST_DATA_SOURCE == "VisTir":
            metrics_per_scene = defaultdict(list)
            for o in outputs:
                metrics_per_scene[o["metrics"]["scene_id"][0]].append(o["metrics"])

            aucs_per_scene = {}
            for scene_id, scene_metrics in metrics_per_scene.items():
                m = {k: flattenList(all_gather(flattenList([_m[k] for _m in scene_metrics]))) for k in scene_metrics[0]}
                aucs_per_scene[scene_id] = aggregate_metrics(m, self.config.TRAINER.EPI_ERR_THR)

            val_metrics_4tb = {f"auc@{thr}": np.mean([aucs_per_scene[s][f"auc@{thr}"] for s in aucs_per_scene]) for thr in (5, 10, 20)}
        else:
            _metrics = [o["metrics"] for o in outputs]
            metrics = {k: flattenList(gather(flattenList([m[k] for m in _metrics]))) for k in _metrics[0]}
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

        # Dump predictions if requested
        if self.dump_dir is not None:
            dumps = flattenList(gather(flattenList([o["dumps"] for o in outputs])))
            np.save(Path(self.dump_dir) / "XoFTR_pred_eval", dumps)
            logger.info(f"Saved predictions + eval to {self.dump_dir}")

        # Rank‑zero prints profiler + metrics
        if self.trainer.is_global_zero:
            print(self.profiler.summary())
            logger.info("\n" + pprint.pformat(val_metrics_4tb))

    # ==================================================
    # Training end hook ensures the metric exists (belt & braces)
    # ==================================================
    def on_train_end(self):
        # If for whatever reason no validation happened in the very last epoch, re‑emit the last known auc@10
        if "auc@10" not in self.trainer.callback_metrics:
            self.log("auc@10", torch.tensor(self.latest_val_auc[10]), prog_bar=False, on_epoch=True, on_step=False)
