from typing import *
import torch as th
import pytorch_lightning as pl
from .utils import *
from lion_pytorch import Lion
from torch import nn, Tensor, optim
import torch.nn.functional as F


class CatMLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        shared_sizes: List[int],
        feature_sizes: List[int],
        final_size: int,
        act: str = "leakyrelu",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.act = act_fn_dict[act]
        shared_sizes = [input_size] + shared_sizes
        dnn_layers = []

        for i in range(len(shared_sizes) - 1):
            dnn_layers.append(nn.Linear(shared_sizes[i], shared_sizes[i + 1]))
            dnn_layers.append(nn.BatchNorm1d(shared_sizes[i + 1]))
            dnn_layers.append(self.act)
            dnn_layers.append(nn.Dropout(dropout))

        self.dnn_layers = nn.Sequential(*dnn_layers)
        self.feature_extractors = nn.ModuleList()
        for feature_size in feature_sizes:
            layers = [
                nn.Linear(shared_sizes[-1], feature_size),
                nn.BatchNorm1d(feature_size),
                self.act,
                nn.Dropout(dropout),
                nn.Linear(feature_size, 1),
            ]
            self.feature_extractors.append(nn.Sequential(*layers))
        self.final_layer = nn.Linear(len(feature_sizes), final_size)

    def forward(self, x):
        mid_output = self.dnn_layers(x)
        features = [extractor(mid_output) for extractor in self.feature_extractors]
        combined_features = th.cat(features, dim=1)
        final_output = self.final_layer(combined_features)
        return combined_features, final_output


class CatNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        shared_sizes: List[int],
        feature_sizes: List[int],
        final_size: int,
        act: str = "leakyrelu",
        dropout: float = 0.3,
        loss_fn: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        loss_weights: List[int] = [1, 1, 1],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.input_size = input_size
        self.feature_sizes = feature_sizes
        self.lr = lr
        self.loss_fn = loss_fn_dict[loss_fn]
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.args = args
        self.model = CatMLPModel(
            input_size=input_size,
            shared_sizes=shared_sizes,
            feature_sizes=feature_sizes,
            final_size=final_size,
            dropout=dropout,
            act=act,
        )
        self._weight_init()
        self.prior_losses = None 

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=1, end_factor=0.1, total_iters=120
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.1
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def _get_reconstruction_loss(self, outputs, y):
        mid_output, last_output = outputs
        loss = th.tensor(0.0, device=y.device)
        # var = y.std(dim=0)
        # print(f"var:{var}")
        for i, weight in enumerate(self.loss_weights):
            loss_i = weight * self.loss_fn(mid_output[:, i], y[:, i])
            # print(f"loss-{i}:{loss_i:.2f}")
            print(loss_i)
            loss += loss_i

        loss += self.loss_fn(last_output.flatten(), y[:, 0])
        # add l2 regularization
        l2_reg = th.tensor(0.0, device=y.device)
        for name, param in self.named_parameters():
            if "weight" in name and "bn" not in name:
                l2_reg += th.norm(param, p=2)
        loss += self.kwargs["l2"] * l2_reg
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self._get_reconstruction_loss(outputs, y)
        met = calc_catmets(outputs, y, "train")
        self.log("train_loss", loss)
        self.log_dict(met)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self._get_reconstruction_loss(outputs, y)
        met = calc_catmets(outputs, y, "valid")
        self.log("val_loss", loss)
        self.log_dict(met)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self._get_reconstruction_loss(outputs, y)
        self.log("test_loss", loss)
        met = calc_catmets(outputs, y, "test")
        self.log_dict(met)
        return loss


def calculate_prior_losses(module, data_loader):
    prior_losses = []
    all_labels = []
    for batch in data_loader:
        _, labels = batch
        all_labels.append(labels)
    all_labels = th.cat(all_labels, dim=0)
    mean_labels = th.mean(all_labels, dim=0)

    for i in range(len(module.loss_weights)):
        prior_loss = th.mean((all_labels[:, i] - mean_labels[i]) ** 2).item()
        prior_losses.append(prior_loss)

    module.prior_losses = prior_losses
