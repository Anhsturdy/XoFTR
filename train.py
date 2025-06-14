import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy  # replaces DDPPlugin
# NOTE: "strategies" is the new path in PL 2.x+

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_xoftr import PL_XoFTR

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=lambda x: bool(strtobool(x)),
                        nargs='?', default=True, help='use pinned memory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--disable_ckpt', action='store_true', help='disable checkpoint saving')
    parser.add_argument('--profiler_name', type=str, default=None, help='[inference, pytorch]')
    parser.add_argument('--parallel_load_data', action='store_true', help='parallel data loading')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--num_sanity_val_steps', type=int, default=10)
    parser.add_argument('--benchmark', type=bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)

    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    profiler = build_profiler(args.profiler_name)
    model = PL_XoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"XoFTR LightningModule initialized!")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"XoFTR DataModule initialized!")

    logger = [TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)]
    ckpt_dir = Path(logger[0].log_dir) / 'checkpoints'
    
    log_dir = logger[0].log_dir  # <- Fix added here
    
    if not config.TRAINER.USE_WANDB:
        logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
        ckpt_dir = Path(logger.log_dir) / 'checkpoints'
        log_dir = logger.log_dir
    else:
        # Use a list if using both TensorBoard and WandB
        logger = [
            TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False),
            WandbLogger(name=args.exp_name + f"_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}", project='XoFTR')
        ]
        ckpt_dir = Path(logger[0].log_dir) / 'checkpoints'
        log_dir = logger[0].log_dir

    ckpt_callback = ModelCheckpoint(
        monitor='auc@10',
        verbose=True,
        save_top_k=-1,
        mode='max',
        save_last=True,
        dirpath=str(ckpt_dir),
        filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        benchmark=True,
        max_epochs=30,
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        num_sanity_val_steps=10,
        limit_val_batches=1.0,
        enable_checkpointing=not args.disable_ckpt,
        default_root_dir=log_dir,
        callbacks=callbacks,
        profiler=None,
        logger=logger
    )

    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)



if __name__ == '__main__':
    main()