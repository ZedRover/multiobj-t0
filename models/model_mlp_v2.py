from typing import *
import torch as th
import pytorch_lightning as pl
from torch import nn, Tensor, optim
from .utils import *
from lion_pytorch import Lion
import torch.nn as nn
import torch.nn.functional as F


class MetaModel(nn.Module):
    def __init__(
        self,
        input_size: int = 101,
        shared_sizes: List[int] = [200, 100],
        tower_sizes: List[List[int]] = [[1], [1], [1]],
        act="leakyrelu",
        dropout=0.3,
    ):

        super().__init__()
        self.act = act_fn_dict[act]
        self.shared_layers = self._make_layers(
            [input_size] + shared_sizes, dropout
        )

        self.towers = nn.ModuleList(
            [
                self._make_layers(
                    [shared_sizes[-1]] + sizes, dropout, False, False, False, False
                )
                for sizes in tower_sizes
            ]
        )

    def forward(self, x):
        x = self.shared_layers(x)
        return [tower(x) for tower in self.towers]

    def _make_layers(
        self,
        layer_sizes,
        dropout,
        batch_norm=True,
        inner_dropout=False,
        final_dropout=True,
        input_dropout=True,
    ):
        layers = []
        for i in range(len(layer_sizes) - 1):
            if i == 0 and input_dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(self.act)
            if inner_dropout:
                layers.append(nn.Dropout(dropout))
        if not final_dropout and inner_dropout:
            layers.pop()
        elif final_dropout and not inner_dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)


class MtNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        tower_sizes: List[List[int]],
        hidden_sizes: List[int] = [12, 12],
        dropout: float = 0,
        act: str = "leakyrelu",
        loss_fn: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        loss_weights: List[int] = [
            0.5,
            0.2,
            0.3,
        ],
        *args: th.Any,
        **kwargs: th.Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.tower_sizes = tower_sizes
        self.lr = lr
        self.loss_fn = loss_fn_dict[loss_fn]
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.args = args
        self.model = MetaModel(
            input_size=input_size,
            shared_sizes=hidden_sizes,
            tower_sizes=tower_sizes,
            dropout=dropout,
            act=act,
        )
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )

    def configure_optimizers(self) -> Any:
        return optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
    def _get_reconstruction_loss(self, outputs, y):
        loss = th.zeros(size=(len(outputs),))
        for i, y_hat in enumerate(outputs):
            y_now = y[:, i].reshape(-1, 1)
            loss[i] = self.loss_weights[i] * self.loss_fn(y_hat, y_now)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self._get_reconstruction_loss(outputs, y)
        sum_loss = th.sum(loss)
        met = calc_mets_v2(outputs, y, "train")
        upload_dict = {f"train/loss_{i}": loss[i] for i in range(len(outputs))}
        upload_dict.update({"train/sum_loss": sum_loss})
        upload_dict.update(met)
        self.log_dict(
            upload_dict,
            on_step=True,
            on_epoch=True,
        )
        return sum_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self._get_reconstruction_loss(outputs, y)
        sum_loss = th.sum(loss)
        met = calc_mets_v2(outputs, y, "valid")
        upload_dict = {
            "valid/loss": sum_loss,
            # "valid/losses": loss,
        }
        upload_dict.update(met)
        self.log_dict(
            upload_dict,
            # on_step=True,
            on_epoch=True,
        )
        return sum_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self._get_reconstruction_loss(outputs, y)
        sum_loss = th.sum(loss)
        ic = calc_ic(outputs[0], y[:, 0].reshape(-1, 1))

        self.log_dict(
            {
                "test/loss": sum_loss,
                # "test/losses": loss,
                "test/ic_ret": ic,
            },
            on_epoch=True,
        )
        return loss, ic

    def forward(self, x):
        return self.model(x)
