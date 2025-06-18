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

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data_pretrain import PretrainDataModule
from src.lightning.lightning_xoftr_pretrain import PL_XoFTR_Pretrain

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=lambda x: bool(strtobool(x)),
                        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path')
    parser.add_argument('--disable_ckpt', action='store_true', help='disable checkpoint saving (useful for debugging).')
    parser.add_argument('--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument('--parallel_load_data', action='store_true', help='load datasets in with multiple processes.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--num_sanity_val_steps', type=int, default=10)
    parser.add_argument('--benchmark', type=bool, default=True)
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes to use')
    parser.add_argument('--max_epochs', type=int, default=15, help='maximum number of epochs')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_XoFTR_Pretrain(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"XoFTR LightningModule initialized!")
    
    # lightning data
    data_module = PretrainDataModule(args, config)
    loguru_logger.info(f"XoFTR DataModule initialized!")
    
    # TensorBoard Logger
    logger = [TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)]
    ckpt_dir = Path(logger[0].log_dir) / 'checkpoints'
    if config.TRAINER.USE_WANDB:
        logger.append(WandbLogger(name=args.exp_name + f"_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
                                project='XoFTR'))
    
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(
    verbose=True,
    save_top_k=-1,
    save_last=True,
    dirpath=str(ckpt_dir),
    filename='{epoch}'
)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer for 1 GPU
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger[0], 
        profiler=profiler,
        max_epochs=args.max_epochs if hasattr(args, "max_epochs") else None,
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
