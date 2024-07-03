from typing import *
import torch as th
import pytorch_lightning as pl
from torch import nn, Tensor, optim
from .utils import *
from lion_pytorch import Lion
import torch.nn as nn
import torch.nn.functional as F


class MtModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size=1,
        hidden_size=(256,),
        dropout: float = 0.3,
        act="LeakyReLU",
    ):
        super(MtModel, self).__init__()
        hidden_size = [input_size] + list(hidden_size)
        dnn_layers = []
        drop_input = nn.Dropout(dropout)
        dnn_layers.append(drop_input)
        hidden_units = input_size
        for i, (_input_dim, hidden_units) in enumerate(
            zip(hidden_size[:-1], hidden_size[1:])
        ):
            fc = nn.Linear(_input_dim, hidden_units)
            if act.lower() == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            elif act == "SiLU":
                activation = nn.SiLU()
            else:
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        fc = nn.Linear(hidden_units, output_size)
        dnn_layers.append(fc)
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output


def metrics_func(yhat, y, stage="train"):
    ret = {}
    for i in range(yhat.shape[1]):
        ret[f"{stage}/loss{i}"] = F.mse_loss(yhat[:, i], y[:, i]).item()
        ret[f"{stage}/corr{i}"] = pearsonr(yhat[:, i], y[:, i])
    return ret


class MtNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        output_size: int,
        dropout: float,
        act: str = "leakyrelu",
        loss_fn: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        loss_weights: List[float] = [0.1, 0.1],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.loss_fn = loss_fn_dict[loss_fn]
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.args = args
        self.model = MtModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            act=act,
            dropout=dropout,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def _get_reconstruction_loss(self, yhat, y):
        n_task = self.output_size
        loss = th.zeros(size=(self.output_size,))
        for i in range(n_task):
            loss[i] = self.loss_fn(yhat[:, i], y[:, i])

        return loss.sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self._get_reconstruction_loss(yhat, y)
        dic = metrics_func(yhat, y)
        self.log_dict(dic)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self._get_reconstruction_loss(yhat, y)
        dic = metrics_func(yhat, y, stage="valid")
        self.log_dict(dic)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.model(x)
        loss = self._get_reconstruction_loss(yhat, y)
        dic = metrics_func(yhat, y, stage="test")
        self.log_dict(dic)
        return loss
