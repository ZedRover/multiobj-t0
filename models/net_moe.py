from .mlp_moe import *
import pytorch_lightning as pl
from .utils import *
from torch import optim
import numpy as np

def mmoe_metric(yhat, y, labels, stage: str = "train"):
    ret = {}
    for i in range(len(labels)):
        ret[f"{stage}/ic_{labels[i]}"] = pearsonr(yhat[:, i], y[:, i].reshape(-1, 1))
    return ret


class MMOE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        shared_size: int,
        num_expert: int,
        expert_sizes: List[int],
        tower_sizes: List[int],
        act: str = "leakyrelu",
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        loss_weights: List[int] = [1, 1, 1],
        loss_fn: str = "mse",
        label_names: List[str] = ["ret", "mean", "var"],
        *args,
        **kwargs,
    ):
        super(MMOE, self).__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.input_size = input_size
        self.lr = lr
        self.loss_fn = loss_fn_dict[loss_fn]
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.args = args

        self.model = MOE(
            input_size=input_size,
            shared_size=shared_size,
            num_expert=num_expert,
            expert_sizes=expert_sizes,
            tower_sizes=tower_sizes,
            num_tasks=len(label_names),
            dropout=dropout,
        )

        self._weight_init()
        self.prior_losses = None
        self.label_names = label_names

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
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=20, eta_min=0.1
        # )
        scheduler = optim.lr_scheduler.LinearLR(optimizer, 1, 0.1, 120)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def _get_reconstruction_loss(self, output, y, stage=None):
        loss = th.tensor(0.0, device=y.device)
        for i, weight in enumerate(self.loss_weights):
            loss_i = self.loss_fn(output[:, i].squeeze(), y[:, i].squeeze())
            norm_i = loss_i / (self.prior_losses[i])
            wght_i = weight * norm_i

            if stage == "train":
                self.log(f"train/loss_{self.label_names[i]}", loss_i)
                self.log(f"train/loss_norm_{self.label_names[i]}", norm_i)
                self.log(f"train/loss_weighted_{self.label_names[i]}", wght_i)
            loss += wght_i

        l2_reg = th.tensor(0.0, device=y.device)
        for name, param in self.model.named_parameters():
            if "weight" in name:
                l2_reg += th.norm(param, p=2)

        loss += self.kwargs["l2"] * l2_reg

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self._get_reconstruction_loss(output, y, stage="train")
        met = mmoe_metric(output, y, self.label_names, "train")
        self.log_dict(met)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self._get_reconstruction_loss(output, y, stage="valid")
        met = mmoe_metric(output, y, self.label_names, "valid")
        self.log_dict(met)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self._get_reconstruction_loss(output, y, stage="test")
        met = mmoe_metric(output, y, self.label_names, "test")
        self.log_dict(met)
        return loss


def calculate_prior_losses(
    module,
    data_loader,
    loss="mse",
):
    prior_losses = []
    all_labels = []
    for batch in data_loader:
        _, labels = batch
        all_labels.append(labels)
    all_labels = th.cat(all_labels, dim=0)
    mean_labels = th.mean(all_labels, dim=0, keepdim=True)
    loss_fn = loss_fn_dict[loss]

    for i in range(len(module.loss_weights)):
        # Ensure the target size matches the input size
        target = mean_labels[:, i].expand_as(all_labels[:, i])
        prior_loss = loss_fn(all_labels[:, i], target).item()
        prior_losses.append(prior_loss)
    print("prior_losses: ", prior_losses)

    module.prior_losses = prior_losses
