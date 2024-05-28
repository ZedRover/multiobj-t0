from typing import *
import torch as th
from torch import Tensor, nn, optim
from time import time
from audtorch.metrics.functional import pearsonr as pearsonr_
from lion_pytorch import Lion


def calc_ic(input, target):
    return pearsonr_(input, target, batch_first=False)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def nanstd(input_tensor: Tensor, dim: int = 0, keepdim: bool = True) -> Tensor:
    return th.sqrt(
        th.nanmean(
            (input_tensor - th.nanmean(input_tensor, dim=dim, keepdim=keepdim)) ** 2
        )
    )


def nanmax(tensor, dim=None, keepdim=False):
    min_value = th.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    max_value = th.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


# def nanstd(tensor, dim=None, keepdim=False):
#     output = nanvar(tensor, dim=dim, keepdim=keepdim)
#     output = output.sqrt()
#     return output


def nanprod(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(1).prod(dim=dim, keepdim=keepdim)
    return output


def nancumprod(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(1).cumprod(dim=dim, keepdim=keepdim)
    return output


def nancumsum(tensor, dim=None, keepdim=False):
    output = tensor.nan_to_num(0).cumsum(dim=dim, keepdim=keepdim)
    return output


def nanargmin(tensor, dim=None, keepdim=False):
    max_value = th.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
    return output


def nanargmax(tensor, dim=None, keepdim=False):
    min_value = th.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
    return output


def pearsonr(yhat: Tensor, y: Tensor, batch_first=False) -> Tensor:
    yhat = yhat.squeeze()
    y = y.squeeze()
    idx = ~th.isnan(y)
    yhat = yhat[idx]
    y = y[idx]
    return pearsonr_(yhat, y, batch_first=batch_first)


class CorrLoss(nn.Module):
    """Pearsonr correlation loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, real: Tensor) -> Tensor:
        mean_y, std_y = th.nanmean(real), nanstd(real)
        mean_yhat, std_yhat = th.nanmean(pred), nanstd(pred)
        if std_y == 0 or std_yhat == 0:
            print("std_y or std_yhat is zero")
            return th.tensor([0.0], device=pred.device)
        ret = 1 - th.mean((pred - mean_yhat) * (real - mean_y) / (std_yhat * std_y))
        return ret if ~th.isnan(ret) else th.tensor([0.0], device=pred.device)


class CorrLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, yhat, y) -> Tensor:
        return pearsonr(yhat, y)


loss_fn_dict: Dict[str, nn.Module] = {
    "mse": nn.MSELoss(reduction="mean"),
    "huber": nn.HuberLoss(delta=10, reduction="mean"),
    "corr": CorrLoss(),
}

act_fn_dict: Dict[str, nn.Module] = {
    "leakyrelu": nn.LeakyReLU(negative_slope=0.1),
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "silu": nn.SiLU(),
}


def calc_mets(outputs, y, stage="train"):
    ret = {}
    for i, (task, y_hat) in enumerate(outputs.items()):
        ret[f"{stage}/ic-{task}"] = pearsonr(y_hat, y[:, i].reshape(-1, 1))

    return ret


def calc_mets_v2(outputs, y, stage="train"):
    ret = {}
    for i, yhat in enumerate(outputs):
        ret[f"{stage}/ic_{i}"] = pearsonr(yhat, y[:, i].reshape(-1, 1))
    return ret


def calc_catmets(outputs, y, stage="train"):
    mid_output, last_output = outputs
    ret = {}
    for i in range(mid_output.shape[1]):
        ret[f"{stage}/ic_{i}"] = pearsonr(mid_output[:, i], y[:, i].reshape(-1, 1))
    ret[f"{stage}/ic_ret"] = pearsonr(last_output, y[:, 0].reshape(-1, 1))
    return ret
