import torch
from torch import Tensor


def huber(x: Tensor, k: int = 1.0) -> Tensor:
    x = x.abs()
    return torch.where(x < k, 0.5 * x.pow(2), k * (x - 0.5 * k))
