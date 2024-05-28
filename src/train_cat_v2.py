import copy
import sutils
import common as cm
from models.mlp_cat_v2 import CatNet
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
from mlutils.data import DataLoaderY

th.set_float32_matmul_precision("medium")

args = ArgumentParser()
args.add_argument("--fold", "-f", type=int, default=0)
args.add_argument("--gpu", "-g", type=int, default=0)
args.add_argument(
    "--batch_size", "-b", type=int, default=100, help="batch size multiplier"
)
args.add_argument("--num_stocks", "-n", type=int, default=2)
args.add_argument("--save", "-s", type=int, default=0)
args.add_argument("--split", "-sp", type=int, default=0)
args.add_argument("--cur", type=int, default=0)
args.add_argument("--fut", type=int, default=60)
args.add_argument("--desc", "-d", type=str, default=None)
args = args.parse_args()


def backtest_stocks(
    model,
    args,
    test_data,
    stk_list,
):
    model.eval()
    model = model.to(th.device(f"cuda:{args.gpu}"))
    if isinstance(test_data, dict):
        test_x_dict = test_data["x"]
        test_y_dict = test_data["y"]
        test_ts_dict = test_data["z"]
    else:
        test_x_dict = test_data.test_x_dict
        test_y_dict = test_data.test_y_dict
        test_ts_dict = test_data.test_ts_dict
    result = []

    with th.no_grad():
        for code in stk_list:
            x_test = test_x_dict[code].to(model.device)
            y_test = test_y_dict[code].detach().numpy()[:, 0]
            y_pred = model(x_test)[1].cpu().detach().numpy()

            ts = test_ts_dict[code]

            pic, bic, qret = cm.daily_performance(y_pred, y_test, ts)
            row_data = {
                "code": code,
                "pIC": pic,
                "bIC": bic,
                **{q: qret[q] for q in cm.QRET_INDEX},
            }
            result.append(row_data)
    model.train()
    return pd.DataFrame(result)


class CustomValidationCallback(pl.Callback):
    def __init__(self, check_n_epoch, datamodule, args, stk_list, logger):
        self.check_n_epoch = check_n_epoch
        self.datamodule = datamodule
        self.args = args
        self.stk_list = stk_list
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epoch = epoch
        self.global_step = trainer.global_step
        if (epoch + 1) % self.check_n_epoch == 0:
            self.perform_validation(pl_module)

    def perform_validation(self, model):
        model.eval()
        test_dict = {
            "x": self.datamodule.test_x_dict,
            "y": self.datamodule.test_y_dict,
            "z": self.datamodule.test_ts_dict,
        }
        result = backtest_stocks(model, self.args, test_dict, self.stk_list)
        self.logger.log_metrics(
            {
                "test/pIC": result.pIC.mean(),
                "test/bIC": result.bIC.mean(),
                "test/q01": result.q01.mean(),
                "test/q99": result.q99.mean(),
            },
            step=self.global_step,
        )


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    stk_list = cm.SELECTED_CODES
    seed = 2022
    if args.num_stocks < 100:
        pl.seed_everything(seed)
        idx = np.random.choice(len(stk_list), size=args.num_stocks, replace=False)
        stk_list = [stk_list[i] for i in idx]
    print(f"selected stk_list:\n{stk_list}")

    ## Model
    labels = [0, 1, 3]

    model_param = {
        "input_size": 101,
        "shared_sizes": [200, 128],
        "feature_sizes": [16, 16, 16],
        "final_size": 1,
        "act": "leakyrelu",
        "dropout": 0.4,
        "lr": 1e-4,
        "loss_fn": "mse",
        "weight_decay": 1e-3,
        "loss_weights": [0.6, 0.2, 0.2],
        "l2": 1e-3,
    }

    model = CatNet(**model_param)
    print(model)
    ## Logger details

    labels_str = "".join(
        [["ret", "mean", "var", "rv", "min", "max", "gap"][i] for i in labels]
    )
    experiment_name = f"f{args.fold}_n{len(stk_list)}_b{args.batch_size}_{args.cur}-{args.fut}_{labels_str}"

    datamodule = sutils.MTDataModule(
        codes=stk_list,
        labels_idx=labels,
        fold=args.fold,
        split=args.split,
        batch_size=args.batch_size * 4000,
        cur=args.cur,
        fut=args.fut,
    )
    logger = wandb_logger.WandbLogger(
        project="Cat" if args.num_stocks == 100 else "Cat-s",
        name=experiment_name,
    )
    logger.experiment.config.update(
        {
            "num_stocks": len(stk_list),
            "tmstamp": timestamp,
            "fold": args.fold,
            "release": False,
            "batch_size": args.batch_size,
            "lables": labels,
            "Notes": args.desc,
        }
    )
    logger.experiment.config.update(model_param)
    ## Training details
    earlystop_callback = EarlyStopping(
        monitor="valid/ic_ret",
        patience=15,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/ic_ret",
        mode="max",
        dirpath=f"./checkpoints/{timestamp}/{args.fold}/",
        filename="model_{valid/ic_ret:.3f}",
        save_top_k=1,
    )
    gas_callback = GradientAccumulationScheduler(scheduling={0: 3, 5: 2, 10: 1})

    test_callback = CustomValidationCallback(10, datamodule, args, stk_list, logger)
    trainer = pl.Trainer(
        devices=[args.gpu],
        callbacks=[
            earlystop_callback,
            checkpoint_callback,
            RichProgressBar(),
            gas_callback,
            test_callback,
        ],
        logger=logger,
        max_epochs=80,
        precision="16-mixed",
        # detect_anomaly=True,
        # fast_dev_run=True,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")
