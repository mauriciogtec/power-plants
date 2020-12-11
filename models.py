from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
import torch


class LaggedSpatialRegression(nn.Module):
    def __init__(
        self,
        units: int,
        kernel_size: int,
        nrows: int,
        ncols: int,
        use_bias: bool = True,
        non_linear: bool = False,
        use_seasonality: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = nn.Parameter(
            0.1 * torch.randn(nrows, ncols, units, kernel_size)
        )
        if non_linear:
            self.W = nn.Parameter(0.2 * torch.randn(nrows, ncols, units))

        self.seasonality = use_seasonality
        if use_seasonality:
            self.W_season = nn.Parameter(
                0.01 * torch.randn(nrows, ncols)
            )

        self.kernel_size = kernel_size
        self.units = units
        self.nrows = nrows
        self.ncols = ncols

        self.non_linear = non_linear
        self.use_bias = use_bias
        if use_bias or non_linear:
            self.alpha = nn.Parameter(torch.zeros(nrows, ncols))
            if non_linear:
                self.alpha0 = nn.Parameter(torch.zeros(nrows, ncols, units))

    def forward(
        self, inputs: Tensor, season: Optional[Tensor] = None
    ) -> Tensor:
        # causal conv
        inputs = inputs.unsqueeze(0)  # expand batch dim
        inputs = F.pad(inputs, (self.kernel_size - 1, 0))

        if not self.non_linear:
            kernel = self.kernel.view(-1, self.units, self.kernel_size)

            if self.use_bias:
                alpha = self.alpha.view(-1)
                out = F.conv1d(inputs, kernel, alpha)
            else:
                out = F.conv1d(inputs, kernel)
            out = out[:, :, (self.kernel_size - 1) :]
        else:
            kernel = self.kernel.view(-1, 1, self.kernel_size)
            alpha0 = self.alpha0.view(-1)
            out = F.conv1d(inputs, kernel, alpha0, groups=self.units)
            out = torch.tanh(out)
            out = out.view(self.nrows * self.ncols, self.units, -1)
            alpha = self.alpha.view(-1, 1)
            W = self.W.view(-1, self.units, 1)
            out = (out * W).sum(1) + alpha
            out = out[:, (self.kernel_size - 1) :]

        out = out.view(self.nrows, self.ncols, -1)
        
        if self.seasonality:
            out = out + self.W_season.unsqueeze(-1) * season

        return out

    def tv_penalty(self, power: int = 2) -> Tensor:
        x = self.kernel
        dr = torch.abs(x[:, :-1, :, :] - x[:, 1:, :, :]).pow(power)
        dc = torch.abs(x[-1:, :, :, :] - x[1:, :, :, :]).pow(power)
        loss = dr.mean() + dc.mean()

        if self.non_linear:
            x = self.W
            dr = torch.abs(x[:, :-1, :] - x[:, 1:, :]).pow(power)
            dc = torch.abs(x[-1:, :, :] - x[1:, :, :]).pow(power)
            loss = loss + dr.mean() + dc.mean()

        return loss

    def kernel_smoothness(self, power: int = 2) -> Tensor:
        loss = 0.0
        x = self.kernel
        dz = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().pow(power)
        loss = dz.mean()
        return loss

    def shrink_penalty(self, power: int = 2) -> Tensor:
        if not self.non_linear:
            loss = self.kernel.norm(dim=-1).pow(power).mean()
        else:
            loss = self.W.abs().pow(power).mean()

        return loss
