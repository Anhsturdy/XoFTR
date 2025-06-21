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
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.config = config
        _config = lower_config(self.config)
        self.xoftr_cfg = lower_config(_config['xoftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        self.matcher = XoFTR(config=_config['xoftr'])
        self.loss = XoFTRLoss(_config)

        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        self.dump_dir = dump_dir

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, closure, **kwargs):
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.global_step / warmup_step) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        compute_supervision_coarse(batch, self.config)
        self.matcher(batch)
        compute_supervision_fine(batch, self.config)
        self.loss(batch)

    def _compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)
        compute_pose_errors(batch, self.config)
        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers']
        }
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            metrics.update({'scene_id': batch['scene_id']})
        return {'metrics': metrics}, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        if self.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            for k, v in batch['loss_scalars'].items():
                self.log(f'train/{k}', v, on_step=True, on_epoch=False, prog_bar=False)
        return {'loss': batch['loss']}

    def on_training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/avg_loss_on_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            self.matcher(batch)
        else:
            self._trainval_inference(batch)
        ret_dict, _ = self._compute_metrics(batch)
        if self.config.DATASET.VAL_DATA_SOURCE != "VisTir":
            ret_dict['loss_scalars'] = batch['loss_scalars']
        return ret_dict

    def on_validation_epoch_end(self, outputs):
        val_metrics_4tb = defaultdict(list)
        if self.config.DATASET.VAL_DATA_SOURCE == "VisTir":
            metrics_per_scene = defaultdict(list)
            for o in outputs:
                scene_id = o['metrics']['scene_id'][0]
                metrics_per_scene[scene_id].append(o['metrics'])

            for scene_id, scene_metrics in metrics_per_scene.items():
                metrics = {k: flattenList(all_gather(flattenList([m[k] for m in scene_metrics]))) for k in scene_metrics[0]}
                auc = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
                for thr in [5, 10, 20]:
                    val_metrics_4tb[f'auc@{thr}'].append(auc[f'auc@{thr}'])
        else:
            metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([m[k] for m in metrics]))) for k in metrics[0]}
            auc = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                val_metrics_4tb[f'auc@{thr}'].append(auc[f'auc@{thr}'])

        for thr in [5, 10, 20]:
            self.log(f'auc@{thr}', torch.tensor(np.mean(val_metrics_4tb[f'auc@{thr}'])), prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.matcher(batch)
        ret_dict, rel_pair_names = self._compute_metrics(batch)
        if self.dump_dir is not None:
            keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf_f', 'epi_errs'}
            pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].shape[0]
            dumps = []
            for b_id in range(bs):
                item = {'pair_names': pair_names[b_id], 'identifier': '#'.join(rel_pair_names[b_id])}
                mask = batch['m_bids'] == b_id
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
        return ret_dict

    def on_test_epoch_end(self, outputs):
        if self.config.DATASET.TEST_DATA_SOURCE == "VisTir":
            metrics_per_scene = defaultdict(list)
            for o in outputs:
                scene_id = o['metrics']['scene_id'][0]
                metrics_per_scene[scene_id].append(o['metrics'])

            aucs_per_scene = {}
            for scene_id, scene_metrics in metrics_per_scene.items():
                metrics = {k: flattenList(all_gather(flattenList([m[k] for m in scene_metrics]))) for k in scene_metrics[0]}
                aucs_per_scene[scene_id] = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

            val_metrics_4tb = {f'auc@{thr}': np.mean([aucs_per_scene[s][f'auc@{thr}'] for s in aucs_per_scene]) for thr in [5, 10, 20]}
        else:
            metrics = {k: flattenList(gather(flattenList([o['metrics'][k] for o in outputs]))) for k in outputs[0]['metrics']}
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            all_dumps = flattenList([o['dumps'] for o in outputs if 'dumps' in o])
            np.save(Path(self.dump_dir) / 'XoFTR_pred_eval.npy', all_dumps)

        if self.global_rank == 0:
            logger.info('\n' + pprint.pformat(val_metrics_4tb))