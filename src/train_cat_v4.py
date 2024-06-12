import copy
import sutils
from sutils import *
import common as cm
from models.mlp_cat_v3 import *
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
args.add_argument(
    "--labels", type=int, nargs="+", default=[0, 1, 3], help="A list of integers"
)
args = args.parse_args()
cur, fut = args.cur, args.fut
other_labels = ["mean", "var", "vol", "min", "max", "gap"]
other_labels = [f"{other_labels[i]}_{cur}-{fut}" for i in args.labels]
label_names = [
    "ret_60",
    "ret_120",
    "ret_180",
    # "ret_300",
    # "ret_600",
    "ret_6000",
] + other_labels


def get_label(code, cur=0, fut=60, label_idx=[]):
    raw_ret = sa.attach(f"label_{code}")
    n = len(raw_ret)

    path = f"/mnt/disk1/multiobj_dataset/{code}"
    labels = []
    for label in label_names:
        labels.append(np.load(f"{path}/{label}.npy").astype(np.float32)[:n])
    res = np.concatenate([raw_ret] + labels, axis=1)
    return res


class MTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        codes: List[str],
        labels_idx: List[int],
        fold: int,
        split: int,
        batch_size: int = 40000,
        cur: int = 60,
        fut: int = 120,
        collate_fn: Callable = None,
        collate_type: str = "pool",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.codes = codes
        self.batch_size = batch_size

        dates = cm.dates
        train_end = dates[fold]
        test_end = dates[fold + 1]
        x_trains, x_valids, x_tests, y_trains, y_valids, y_tests = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        test_x_dict = {}
        test_y_dict = {}
        test_ts_dict = {}
        dates = cm.dates
        train_end = dates[fold]
        test_end = dates[fold + 1]

        for code in tqdm(codes):
            df, y, ts = sutils.get_data(code)
            label = get_label(code, cur, fut, labels_idx)

            train_idx = (ts < train_end) & (ts >= train_end - 10000)
            test_idx = (ts < test_end) & (ts >= train_end)
            x_train, x_test = df[train_idx, :], df[test_idx, :]
            y_train, y_test = label[train_idx, :], label[test_idx, :]

            x_train, x_test, y_train, y_test = map(
                th.from_numpy, (x_train, x_test, y_train, y_test)
            )

            idx = list(range(len(y_train)))

            if split == 0:
                train_idx, valid_idx = train_test_split(
                    idx,
                    test_size=0.3,
                    shuffle=True,
                )

            elif split == 1:
                split = int(len(idx) * 0.7)
                train_idx, valid_idx = idx[:split], idx[split:]

            else:
                split = int(len(idx) * 0.7)
                gap = int(len(idx) * 0.01)
                train_idx, valid_idx = idx[:split], idx[split + gap :]

            x_train, x_valid, y_train, y_valid = (
                x_train[train_idx],
                x_train[valid_idx],
                y_train[train_idx],
                y_train[valid_idx],
            )

            # quantile = 0.03
            # lb, ub = th.quantile(y_train, quantile, dim=0), th.quantile(
            #     y_train, 1 - quantile, dim=0
            # )
            # y_train = th.clamp(y_train, lb, ub)

            x_trains.append(x_train)
            x_valids.append(x_valid)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_valids.append(y_valid)
            y_tests.append(y_test)
            test_x_dict[code] = x_test
            test_y_dict[code] = y_test
            ts_test = ts[test_idx]
            test_ts_dict[code] = ts_test
        x_train, x_valid, x_test, y_train, y_valid, y_test = map(
            th.vstack, (x_trains, x_valids, x_tests, y_trains, y_valids, y_tests)
        )
        self.train_dataset = (x_train, y_train)
        self.valid_dataset = (x_valid, y_valid)
        self.test_dataset = (x_test, y_test)
        self.test_x_dict = test_x_dict
        self.test_y_dict = test_y_dict
        self.test_ts_dict = test_ts_dict

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoaderY(
            *self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoaderY(*self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoaderY(*self.test_dataset, batch_size=self.batch_size)


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
                "test/qret": (result.q99.mean() - result.q01.mean()) / 2,
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
    labels = label_names

    model_param = {
        "input_size": 101,
        "shared_sizes": [200, 128],
        "feature_sizes": [16] * len(labels),
        "final_size": 1,
        "act": "leakyrelu",
        "dropout": 0.4,
        "lr": 1e-4,
        "loss_fn": "mse",
        "weight_decay": 1e-3,
        "loss_weights": (
            [0.6, 0.4, 0.1] + [0.4] * (len(labels) - 3)
            if len(labels) > 3
            else [0.6, 0.4, 0.1]
        ),
        "l2": 0,
    }

    model = CatNet(**model_param)
    print(model)
    ## Logger details

    labels_str = "+".join(label_names)

    experiment_name = f"f{args.fold}-{args.cur}-{args.fut}_{len(labels)}"

    datamodule = MTDataModule(
        codes=stk_list,
        labels_idx=labels,
        fold=args.fold,
        split=args.split,
        batch_size=args.batch_size * 4000,
        cur=args.cur,
        fut=args.fut,
    )
    logger = wandb_logger.WandbLogger(
        project="MTL-CAT-V4" if args.num_stocks == 100 else "Cat-s",
        name=experiment_name,
    )
    logger.experiment.config.update(
        {
            "num_stocks": len(stk_list),
            "tmstamp": timestamp,
            "fold": args.fold,
            "batch_size": args.batch_size,
            "labels": labels_str,
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
        max_epochs=120,
        precision="16-mixed",
        # detect_anomaly=True,
        # fast_dev_run=True,
    )
    calculate_prior_losses(
        model, datamodule.train_dataloader(), loss=model_param["loss_fn"]
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path="best")
