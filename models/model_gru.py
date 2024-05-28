from typing import *
import torch as th
from torch import nn, optim, Tensor
from .utils import *
import pytorch_lightning as pl


class GRUModel(nn.Module):
    def __init__(
        self, input_size=6, output_size=1, hidden_size=64, num_layers=2, dropout=0.0
    ):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.input_size = input_size

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GRUNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 1e-3,
        loss_fn: str = "mse",
        weight_decay: float = 1e-3,
        *args: th.Any,
        **kwargs: th.Any
    ):
        super().__init__()
        self.kwargs = kwargs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.loss_fn = loss_fn_dict[loss_fn]
        self.weight_decay = weight_decay

        self.model = GRUModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return Lion(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _get_reconstruction_loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        yhat = self(x)
        loss = self._get_reconstruction_loss(yhat, y)
        # self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(
            {"train_loss": loss, "train_ic": calc_ic(yhat, y)},
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        yhat = self(x)
        loss = self._get_reconstruction_loss(yhat, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_ic": calc_ic(yhat, y),
            },
            on_epoch=True,
            on_step=True,
            # prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        yhat = self(x)
        loss = self._get_reconstruction_loss(yhat, y)
        ic = calc_ic(yhat, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_ic": calc_ic(yhat, y),
            },
        )
        return loss, ic


def generate_tsdata(data, seq_len):
    # 假设 data 的形状为 (N, F)，其中 N 是数据点的数量，F 是特征的数量
    N, F = data.shape
    # 在数据的开头填充 seq_len-1 行零，以确保所有行都有足够的历史数据
    padded_data = nn.functional.pad(
        data, (0, 0, seq_len - 1, 0), mode="constant", value=0
    )
    # 使用 unfold 函数创建滑动窗口视图
    unfold_data = padded_data.unfold(dimension=0, size=seq_len, step=1)
    # 调整维度顺序为 (N, seq_len, F)
    timeseries_data = unfold_data.permute(0, 2, 1)
    return timeseries_data
