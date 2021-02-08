from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from torch import nn


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
        ar_term: Optional[Tensor] = None,
        positive_kernel: bool = False,
        positive_bias: bool = False,
        covariates: Optional[Tensor] = None
    ) -> None:
        super().__init__()
        if positive_kernel:
            self.kernel = nn.Parameter(
                (torch.randn(nrows, ncols, units, kernel_size) * 0.01).exp()
            )
        else:
            self.kernel = nn.Parameter(
                torch.randn(nrows, ncols, units, kernel_size).exp()
            )
        self.kernel_positive = None
        if non_linear:
            self.W = nn.Parameter(0.2 * torch.randn(nrows, ncols, units))

        self.seasonality = use_seasonality
        if use_seasonality:
            self.W_season = nn.Parameter(
                0.01 * torch.randn(nrows, ncols)
            )
            # self.W_season = nn.Parameter(
            #     0.01 * torch.randn(1, 1)
            # )
        self.ar_term = ar_term
        if ar_term is not None:
            ardim = ar_term.shape[-1]
            self.W_ar_term = nn.Parameter(
                0.01 * torch.randn(nrows, ncols, 1, ardim)
            )

        self.kernel_size = kernel_size
        self.units = units
        self.nrows = nrows
        self.ncols = ncols

        self.non_linear = non_linear
        self.use_bias = use_bias
        if use_bias or non_linear:
            self.alpha = nn.Parameter(torch.zeros(nrows, ncols))
            self.alpha_positive = None
            if non_linear:
                self.alpha0 = nn.Parameter(torch.zeros(nrows, ncols, units))
        self.positive_kernel = positive_kernel
        self.positive_bias = positive_bias

        self.covariates = covariates
        if self.covariates is not None:
            self.covariates_nn = nn.Sequential(

            )

    def forward(
        self, inputs: Tensor, season: Optional[Tensor] = None
    ) -> Tensor:
        # causal conv
        inputs = inputs.unsqueeze(0)  # expand batch dim
        inputs = F.pad(inputs, (self.kernel_size - 1, 0))

        if not self.non_linear:
            if self.positive_kernel:
                # kernel = self.huber(self.kernel, .05)  #
                # kernel = F.softplus(self.kernel)  #
                kernel = self.kernel.exp()
                self.kernel_positive = kernel
                # kernel = F.relu(self.kernel)
            else:
                kernel = self.kernel
            kernel = kernel.view(-1, self.units, self.kernel_size)

            if self.use_bias:
                if self.positive_bias:
                    # alpha = self.huber(self.alpha, .05)  #
                    # alpha = F.softplus(self.alpha)
                    alpha = self.alpha.exp()
                    self.alpha_positive = alpha
                else:
                    alpha = self.alpha
                alpha = alpha.view(-1)
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

        if self.ar_term is not None:
            phi = torch.tanh(self.W_ar_term)
            out2 = (self.ar_term * phi).sum(-1)
            out = out + out2

        return out

    def tv_penalty(self, power: int = 2) -> Tensor:
        if self.positive_kernel:
            x = self.kernel_positive
            # x = F.relu(self.kernel)
        else:
            x = self.kernel
        dr = torch.abs(x[:, :-1, :, :] - x[:, 1:, :, :]).pow(power)
        dc = torch.abs(x[:-1, :, :, :] - x[1:, :, :, :]).pow(power)
        loss = dr.mean() + dc.mean()

        if self.non_linear:
            x = self.W
            dr = torch.abs(x[:, :-1, :] - x[:, 1:, :]).pow(power)
            dc = torch.abs(x[:-1, :, :] - x[1:, :, :]).pow(power)
            loss = loss + dr.mean() + dc.mean()

        return loss

    def kernel_smoothness(self, power: int = 2) -> Tensor:
        loss = 0.0
        if self.positive_kernel:
            # x = F.relu(self.kernel)
            x = self.kernel_positive
        else:
            x = self.kernel
        dz = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().pow(power)
        loss = dz.mean()
        return loss

    def huber(self, x: Tensor, k: int = 1.0) -> Tensor:
        x = x.abs()
        return torch.where(x < k, 0.5 * x.pow(2), k * (x - 0.5 * k))

    def shrink_penalty(self, power: int = 2) -> Tensor:
        if not self.non_linear:
            # if self.positive_kernel:
            #     loss = self.kernel_positive.norm(dim=-1)
            # else:
            #     loss = self.kernel.norm(dim=-1)
            if self.positive_kernel:
                # ker = self.kernel - 0.000001  # shrink to small const
                ker = self.kernel_positive
            else:
                ker = self.kernel
            loss = ker.norm(dim=-1)
            if power == 1:
                loss = self.huber(loss, k=0.01).mean()
            else:
                loss = loss.pow(power).mean()
        else:
            loss = self.W.abs().pow(power).mean()

        return loss
