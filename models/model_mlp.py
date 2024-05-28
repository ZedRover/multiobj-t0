from typing import *
import torch as th
import pytorch_lightning as pl
from torch import nn, Tensor, optim
from .utils import *
from lion_pytorch import Lion
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskMLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        shared_sizes: list[int],
        tower_sizes: dict[str, list[int]],
        act: str = "leakyrelu",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.act = act_fn_dict[act]
        self.shared_layers = self._make_layers(
            [input_size] + shared_sizes,
            dropout,
            batch_norm=True,
            inner_dropout=True,
            final_dropout=True,
        )
        self.towers = nn.ModuleDict(
            {
                task: self._make_layers(
                    [shared_sizes[-1]] + sizes,
                    dropout,
                    batch_norm=False,
                    inner_dropout=False,
                    final_dropout=False,
                    input_dropout=False,
                )
                for task, sizes in tower_sizes.items()
            }
        )

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

    def forward(self, x):
        shared_output = self.shared_layers(x)
        outputs = {task: tower(shared_output) for task, tower in self.towers.items()}
        return outputs


class Net(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        tower_sizes: dict[str, list[int]],
        hidden_sizes: List[int] = [12, 12],
        dropout: float = 0,
        act: str = "leakyrelu",
        loss_fn: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        loss_weights: List[int] = [
            0.1,
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

        self.model = MultiTaskMLPModel(
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
        # return optim.AdamW(
        #     self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        optimizer = Lion(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=0.5, total_iters=10
        # )
        # return [optimizer], [scheduler]
        return optimizer

    def forward(self, x):
        return self.model(x)

    def _get_reconstruction_loss(self, outputs, y):
        loss = th.zeros(size=(len(outputs),))
        for i, (task, y_hat) in enumerate(outputs.items()):
            y_now = y[:, i].reshape(-1, 1)
            loss[i] = self.loss_weights[i] * self.loss_fn(y_hat, y_now)
        # add l2 reguization 
        l2_reg = th.tensor(0.0)
        for param in self.parameters():
            l2_reg += th.norm(param)
        loss = th.sum(loss) + self.kwargs["l2"] * l2_reg
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self._get_reconstruction_loss(outputs, y)
        sum_loss = th.sum(loss)
        met = calc_mets(outputs, y, "train")
        upload_dict = {
            "train/loss": sum_loss,
            # "train/losses": loss,
        }
        upload_dict.update(met)
        self.log_dict(
            upload_dict,
            # on_step=True,
            on_epoch=True,
        )
        return sum_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self._get_reconstruction_loss(outputs, y)
        sum_loss = th.sum(loss)
        met = calc_mets(outputs, y, "valid")
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
        ic = calc_ic(outputs["ret"], y[:, 0].reshape(-1, 1))

        self.log_dict(
            {
                "test/loss": sum_loss,
                # "test/losses": loss,
                "test/ic": ic,
            },
            on_epoch=True,
        )
        return loss, ic