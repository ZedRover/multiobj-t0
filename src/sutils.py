import sys

sys.path.append("../")
sys.path.append("/home/ywang/workspace/valuation_project/src")
sys.path.append("/home/ywang/workspace/valuation_project/")
from typing import *
import pytorch_lightning as pl
import torch as th
import pandas as pd
import numpy as np
import SharedArray as sa
import common as cm
from tqdm import tqdm
from rich.progress import track as tqdm
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch import Tensor
from time import time
import torch.distributed as dist
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from prefetch_generator import BackgroundGenerator


# class DataLoaderX(data.DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__(), max_prefetch=-1)


def get_data(code):
    x = sa.attach(f"factor_{code}")
    y = sa.attach(f"label_{code}")
    z = sa.attach(f"timestamp_{code}")
    return x, y, z


class DataLoaderX:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, **kwargs):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class DataLoaderY:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because DataLoader grabs individual indices of
    the dataset and calls cat (slow).
    This version integrates BackgroundGenerator for multi-threaded prefetching.
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, max_prefetch=-1):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param max_prefetch: the number of batches to prefetch in background.
        :returns: A FastTensorDataLoader.
        """
        assert all(
            t.shape[0] == tensors[0].shape[0] for t in tensors
        ), "All tensors must have the same size in the first dimension."
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_prefetch = max_prefetch

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def _generate_batches(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            shuffled_tensors = [t[r] for t in self.tensors]
        else:
            shuffled_tensors = self.tensors

        start = 0
        while start < self.dataset_len:
            end = min(start + self.batch_size, self.dataset_len)
            yield tuple(t[start:end] for t in shuffled_tensors)
            start = end

    def __iter__(self):
        return BackgroundGenerator(
            self._generate_batches(), max_prefetch=self.max_prefetch
        )

    def __len__(self):
        return self.n_batches


class DataLoaderZ:
    def __init__(
        self,
        *tensors,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        max_prefetch=-1,
    ):
        assert all(
            t.shape[0] == tensors[0].shape[0] for t in tensors
        ), "All tensors must have the same size in the first dimension."

        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_prefetch = max_prefetch

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def _generate_batches(self, tensors):
        start = 0
        while start < self.dataset_len:
            end = min(start + self.batch_size, self.dataset_len)
            yield tuple(t[start:end] for t in tensors)
            start = end

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            shuffled_tensors = [t[r] for t in self.tensors]
        else:
            shuffled_tensors = self.tensors
        if self.num_workers == 0:
            return BackgroundGenerator(
                self._generate_batches(shuffled_tensors), max_prefetch=self.max_prefetch
            )
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                return BackgroundGenerator(
                    executor.map(lambda x: x, self._generate_batches(shuffled_tensors)),
                    max_prefetch=self.max_prefetch,
                )


# {'var_1-60', 'vol_120-180', 'vol_1-60', 'mean_1-60', 'var_60-120', 'mean_60-120', 'mid_price', 'vol_60-120', 'var_120-180', 'mean_120-180'}
def get_label(code, cur=0, fut=60):
    tag = f"{cur}-{fut}"
    path = f"/mnt/disk1/multiobj_dataset/{code}"
    mean, var, vol = (
        np.load(f"{path}/mean_{tag}.npy"),
        np.load(f"{path}/var_{tag}.npy"),
        np.load(f"{path}/vol_{tag}.npy"),
    )
    min, max, gap = (
        np.load(f"{path}/min_{tag}.npy"),
        np.load(f"{path}/max_{tag}.npy"),
        np.load(f"{path}/gap_{tag}.npy"),
    )
    return (
        mean.astype(np.float32),
        var.astype(np.float32),
        vol.astype(np.float32),
        min.astype(np.float32),
        max.astype(np.float32),
        gap.astype(np.float32),
    )


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
            df, y, ts = get_data(code)
            new_labels = get_label(code, cur, fut)
            n = len(y)
            all_labels = [y] + [i[:n] for i in new_labels]
            label = np.concatenate(
                [all_labels[i] for i in labels_idx],
                axis=1,
            )

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
