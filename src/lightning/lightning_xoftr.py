
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
import torch.nn.functional as F

class PL_XoFTR(pl.LightningModule):
    """Lightning wrapper that supports knowledge‑distillation between a
    *teacher* XoFTR and a lighter *student* model with smaller hidden size.
    """
    def __init__(
        self,
        config,
        *,
        pretrained_ckpt: str | None = None,
        profiler: PassThroughProfiler | None = None,
        dump_dir: str | None = None,
        teacher_cfg: dict | None = None,
        teacher_ckpt: str | None = None,
        distill_alpha: float = 0.5,
        distill_temp: float = 1.0,
        student_dim: int = 128,
        teacher_dim: int = 256,
    ):
        super().__init__()
        self.config = config
        _config = lower_config(config)
        self.xoftr_cfg = lower_config(_config["xoftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.dump_dir = dump_dir
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        self.matcher = XoFTR(config=_config["xoftr"])
        self.criterion = XoFTRLoss(_config)

        if pretrained_ckpt:
            state = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            self.matcher.load_state_dict(state, strict=False)
            logger.info(f"Loaded pretrained weights from {pretrained_ckpt}")

        self.distillation = teacher_cfg is not None and teacher_ckpt is not None
        self.distill_alpha = float(distill_alpha)
        self.distill_temp = float(distill_temp)

        if self.distillation:
            from src_teacher.xoftr import XoFTR as TeacherXoFTR
            from src_teacher.config.default import get_cfg_defaults as get_teacher_cfg_defaults
            from src_teacher.utils.misc import lower_config as lower_teacher_config

            _teacher_cfg = lower_teacher_config(get_teacher_cfg_defaults(inference=True))
            self.teacher = TeacherXoFTR(config=_teacher_cfg["xoftr"])
            state = torch.load(teacher_ckpt, map_location="cpu")["state_dict"]
            self.teacher.load_state_dict(state, strict=False)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            self.teacher = None

        if self.distillation and student_dim != teacher_dim:
            self.logit_proj0 = torch.nn.Linear(student_dim, teacher_dim, bias=False)
            self.logit_proj1 = torch.nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.logit_proj0 = torch.nn.Identity()
            self.logit_proj1 = torch.nn.Identity()

    def setup(self, stage: str):
        if isinstance(self.logger, list):
            self.tb_logger = self.logger[0]
            self.wandb_logger = self.logger[1] if len(self.logger) > 1 else None
        else:
            self.tb_logger = self.logger
            self.wandb_logger = None

    def configure_optimizers(self):
        optim = build_optimizer(self, self.config)
        sched = build_scheduler(self.config, optim)
        return [optim], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
        with self.profiler.profile("Compute losses"):
            self.criterion(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Compute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics.update({'scene_id': batch['scene_id']})
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch: dict, batch_idx: int):
        self._trainval_inference(batch)
        loss = batch["loss"]

        if self.distillation:
            with torch.no_grad():
                t_out0, t_out1 = self.teacher(batch, return_logits=True)
            s_out0, s_out1 = self.matcher(batch, return_logits=True)
            s_out0 = self.logit_proj0(s_out0)
            s_out1 = self.logit_proj1(s_out1)
            kl_0 = F.kl_div(F.log_softmax(s_out0 / self.distill_temp, dim=-1), F.softmax(t_out0 / self.distill_temp, dim=-1), reduction="batchmean")
            kl_1 = F.kl_div(F.log_softmax(s_out1 / self.distill_temp, dim=-1), F.softmax(t_out1 / self.distill_temp, dim=-1), reduction="batchmean")
            distill_loss = (kl_0 + kl_1) / 2.0 * (self.distill_temp ** 2)
            loss = (1.0 - self.distill_alpha) * loss + self.distill_alpha * distill_loss
            batch["loss"] = loss
            batch["loss_scalars"].update({"distill_loss": distill_loss.detach()})

        if self.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in batch["loss_scalars"].items():
                self.tb_logger.experiment.add_scalar(f"train/{k}", v, self.global_step)
                if self.config.TRAINER.USE_WANDB and self.wandb_logger:
                    self.wandb_logger.log_metrics({f"train/{k}": v}, self.global_step)

        return {"loss": loss}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {k: v for k, v in state.items() if not k.startswith("teacher.")}

    # ------------------------------------------------------------------
    # Epoch‑end helpers (unchanged except cosmetic edits)...
    # ------------------------------------------------------------------
    # def on_training_epoch_end(self):
    #     if self.global_rank == 0:
    #         avg = torch.stack(self.training_step_outputs).mean()
    #         self.tb_logger.experiment.add_scalar(
    #             "train/avg_loss_on_epoch", avg, global_step=self.current_epoch
    #         )
    #         if self.config.TRAINER.USE_WANDB and self.wandb_logger:
    #             self.wandb_logger.log_metrics({"train/avg_loss_on_epoch": avg}, self.current_epoch)
    #     self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            with self.profiler.profile("XoFTR"):
                self.matcher(batch)
        else:
            self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE, ret_dict=ret_dict)

        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            out = {**ret_dict, 'figures': figures}
        else:
            out = {**ret_dict, 'loss_scalars': batch['loss_scalars'], 'figures': figures}

        self.validation_step_outputs.append(out)
        return out
        
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = -1 if self.trainer.sanity_checking else self.current_epoch
            
            if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
                metrics_per_scene = {}
                for o in outputs:
                    if not o['metrics']['scene_id'][0] in metrics_per_scene.keys():
                        metrics_per_scene[o['metrics']['scene_id'][0]] = []
                    metrics_per_scene[o['metrics']['scene_id'][0]].append(o['metrics'])

                aucs_per_scene = {}
                for scene_id in metrics_per_scene.keys():
                    # 2. val metrics: dict of list, numpy
                    _metrics = metrics_per_scene[scene_id]
                    metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
                    # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
                    val_metrics = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                    aucs_per_scene[scene_id] = val_metrics

                # average the metrics of scenes
                # since the number of images in each scene is different
                val_metrics_4tb = {}
                for thr in [5, 10, 20]:
                    temp = []
                    for scene_id in metrics_per_scene.keys():
                        temp.append(aucs_per_scene[scene_id][f'auc@{thr}'])
                    val_metrics_4tb[f'auc@{thr}'] = float(np.array(temp, dtype=float).mean())
                temp = []
                for scene_id in metrics_per_scene.keys():
                    temp.append(aucs_per_scene[scene_id][f'prec@{self.config.TRAINER.EPI_ERR_THR:.0e}'])
                val_metrics_4tb[f'prec@{self.config.TRAINER.EPI_ERR_THR:.0e}'] = float(np.array(temp, dtype=float).mean())
            else:
                # 1. loss_scalars: dict of list, on cpu
                _loss_scalars = [o['loss_scalars'] for o in outputs]
                loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

                # 2. val metrics: dict of list, numpy
                _metrics = [o['metrics'] for o in outputs]
                metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
                # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                if self.config.DATASET.VAL_DATA_SOURCE != "VisTir":
                    for k, v in loss_scalars.items():
                        mean_v = torch.stack(v).mean()
                        self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                    for k, v in val_metrics_4tb.items():
                        self.tb_logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                        if self.config.TRAINER.USE_WANDB and self.wandb_logger:
                            self.wandb_logger.log_metrics({f"metrics_{valset_idx}/{k}": v}, cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.tb_logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf_f', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    if self.config.DATASET.TEST_DATA_SOURCE == "VisTir":
                        item['scene_id'] = batch['scene_id']
                        item['K0'] = batch['K0'][b_id].cpu().numpy()
                        item['K1'] = batch['K1'][b_id].cpu().numpy()
                        item['dist0'] = batch['dist0'][b_id].cpu().numpy()
                        item['dist1'] = batch['dist1'][b_id].cpu().numpy()
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        self.test_step_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        if self.config.DATASET.TEST_DATA_SOURCE == "VisTir":
            # metrics: dict of list, numpy
            metrics_per_scene = {}
            for o in outputs:
                if not o['metrics']['scene_id'][0] in metrics_per_scene.keys():
                    metrics_per_scene[o['metrics']['scene_id'][0]] = []
                metrics_per_scene[o['metrics']['scene_id'][0]].append(o['metrics'])

            aucs_per_scene = {}
            for scene_id in metrics_per_scene.keys():
                # 2. val metrics: dict of list, numpy
                _metrics = metrics_per_scene[scene_id]
                metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
                # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
                val_metrics = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                aucs_per_scene[scene_id] = val_metrics
            
            # average the metrics of scenes
            # since the number of images in each scene is different
            val_metrics_4tb = {}
            for thr in [5, 10, 20]:
                temp = []
                for scene_id in metrics_per_scene.keys():
                    temp.append(aucs_per_scene[scene_id][f'auc@{thr}'])
                val_metrics_4tb[f'auc@{thr}'] = np.array(temp, dtype=float).mean()
        else:
            # metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'XoFTR_pred_eval', dumps)
        self.test_step_outputs.clear()

