import copy
import sutils
from sutils import *
import common as cm
from models.mlp_simple import *
import numpy as np
import pandas as pd
import SharedArray as sa
import torch as th
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    GradientAccumulationScheduler,
    RichProgressBar,
)
from pytorch_lightning.loggers import wandb as wandb_logger
import pytorch_lightning as pl
from functools import lru_cache
from rich.progress import track as tqdm
from torch.utils import data
import os
from argparse import ArgumentParser
import wandb
from datetime import datetime
import sys

sys.path.append("/home/ywang/workspace/valuation_project/src")
from s_utils import ExDataModule


args = ArgumentParser()
args.add_argument("--fold", "-f", type=int, default=3)
args.add_argument("--gpu", "-g", type=int, default=0)
args.add_argument(
    "--batch_size", "-b", type=int, default=100, help="batch size multiplier"
)
args.add_argument("--split", "-sp", type=int, default=0)
args.add_argument("--desc", "-d", type=str, default=None)
args = args.parse_args()


stk_list = cm.SELECTED_CODES

model_param = {
    "input_size": 101,
    "hidden_size": [200, 128],
    "output_size": 2,
    "dropout": 0.3,
}


model = MtNet(**model_param)

datamodule = ExDataModule(
    codes=stk_list, fold=args.fold, batch_size=args.batch_size * int(4e3), split=args.split
)

timestamp = datetime.now().strftime("%y%m%d%H%M")
logger = wandb_logger.WandbLogger(project="WithExr")
logger.experiment.config.update(
    {
        "timestamp": timestamp,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "split": args.split,
        "Notes": args.desc,
    }
)
logger.experiment.config.update(model_param)


earlystop_callback = EarlyStopping(
    monitor="valid/corr0",
    patience=15,
    mode="max",
)
checkpoint_callback = ModelCheckpoint(
    monitor="valid/corr0",
    mode="max",
    dirpath=f"./checkpoints/{timestamp}/{args.fold}/",
    filename="model_{valid/corr0:.3f}",
    save_top_k=1,
)
gas_callback = GradientAccumulationScheduler(scheduling={0: 3, 5: 2, 10: 1})
trainer = pl.Trainer(
    devices=[args.gpu],
    callbacks=[
        earlystop_callback,
        checkpoint_callback,
        RichProgressBar(),
        gas_callback,
    ],
    logger=logger,
    max_epochs=120,
    precision="16-mixed",
    detect_anomaly=True,
)
trainer.fit(model, datamodule)
trainer.test(model, datamodule, ckpt_path="best")
