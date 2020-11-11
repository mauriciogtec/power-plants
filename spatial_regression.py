import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import os
import matplotlib.pyplot as plt


class LaggedSpatialRegression(nn.Module):
    def __init__(
        self,
        units: int,
        kernel_size: int,
        power: int,
        nrows: int,
        ncols: int,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.logits_mu = nn.Parameter(torch.randn(nrows, ncols, units))
        self.loglam = nn.Parameter(1 + torch.randn(nrows, ncols, units))
        self.loggam = nn.Parameter(torch.randn(nrows, ncols, units))
        t = torch.arange(kernel_size, dtype=torch.float32)
        self.t = nn.Parameter(t, requires_grad=False)
        self.kernel_size = kernel_size
        self.power = power
        self.units = units
        self.nrows = nrows
        self.ncols = ncols
        self.use_bias = use_bias
        if use_bias:
            self.alpha = nn.Parameter(torch.randn(nrows, ncols))

    def get_pars(self) -> Tensor:
        mu = self.kernel_size * torch.sigmoid(self.logits_mu)
        gam = F.softplus(self.loggam)
        lam = 1.0 + F.softplus(self.loglam)
        return mu, gam, lam

    def forward(self, inputs: Tensor) -> Tensor:
        mu, gam, lam = self.get_pars()
        mu = mu.view(-1, self.units, 1)
        gam = gam.view(-1, self.units, 1)
        lam = lam.view(-1, self.units, 1)
        t = self.t.view(1, 1, -1)
        kernel = gam * torch.exp(-torch.abs((t - mu) / lam) ** self.power)

        # causal conv
        inputs = inputs.unsqueeze(0)  # expand batch dim
        inputs = F.pad(inputs, (self.kernel_size - 1, 0))

        if self.use_bias:
            alpha = self.alpha.view(-1)
            out = F.conv1d(inputs, kernel, alpha)
        else:
            out = F.conv1d(inputs, kernel)

        out = out[:, :, (self.kernel_size - 1) :]
        out = out.view(self.nrows, self.ncols, -1)

        return out

    def tv_penalty(self, power: int = 2) -> Tensor:
        mu, gam, lam = self.get_pars()
        loss = 0.0
        for par in (lam, gam, mu):
            dr = torch.abs(par[:, :-1, :] - par[:, 1:, :]).pow(power)
            dc = torch.abs(par[:, :, :-1] - par[:, :, 1:]).pow(power)
            loss += dr.mean() + dc.mean()

        return loss

    def shrink_penalty(self, power: int = 2) -> Tensor:
        mu, gam, lam = self.get_pars()
        loss = (
            gam.pow(power).mean()
            # + 0.1 * mu.pow(power).mean()
            # + lam.mean()
        )
        return loss


class LaggedSpatialRegression2(nn.Module):
    def __init__(
        self,
        units: int,
        kernel_size: int,
        power: int,
        nrows: int,
        ncols: int,
        use_bias: bool = True,
        non_linear: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = nn.Parameter(
            0.1 * torch.randn(nrows, ncols, units, kernel_size)
        )
        if non_linear:
            self.W = nn.Parameter(0.2 * torch.randn(nrows, ncols, units))

        self.kernel_size = kernel_size
        self.power = power
        self.units = units
        self.nrows = nrows
        self.ncols = ncols

        self.non_linear = non_linear
        self.use_bias = use_bias
        if use_bias or non_linear:
            self.alpha = nn.Parameter(torch.zeros(nrows, ncols))
            if non_linear:
                self.alpha0 = nn.Parameter(torch.zeros(nrows, ncols, units))

    def forward(self, inputs: Tensor) -> Tensor:
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


def train(
    what: str,
    use_bias: bool,
    free_kernel: bool = False,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    reg=1.0,
    non_linear: bool = False
) -> None:
    data = np.load(f"data/simulation/{what}.npz")
    X_ = data["power_plants"]
    y_ = data["states"]
    # sigs = data["sigs"]
    locs = data["locs"]

    if use_log:
        y_ = np.log(y_ + 1)
        X_ = np.log(X_ + 1)

    if normalize_inputs:
        scale = X_.std(1)
        X_ /= np.expand_dims(scale, -1)

    # std_y_ = y_.std()
    # locs = data["locs"]

    X = torch.FloatTensor(X_).cuda()
    y = torch.FloatTensor(y_).cuda()

    kernel_size = 12
    units, t = X.shape
    y = y[:, :, (kernel_size - 1) :]
    nrows, ncols, _ = y.shape

    if free_kernel:
        model_fun = LaggedSpatialRegression2
    else:
        model_fun = LaggedSpatialRegression

    model = model_fun(
        units=units,
        kernel_size=kernel_size,
        power=2,
        nrows=nrows,
        ncols=ncols,
        use_bias=use_bias,
        non_linear=non_linear
    ).cuda()

    decay = 0.8
    decay_every = 1000
    max_steps = 50000

    optim = torch.optim.Adam(model.parameters(), init_lr, (0.9, 0.99), eps=1e-3)

    eta_ridge = 0.02 * reg
    eta_kernel_smoothness = 0.1
    eta_tv = 0.02 * reg

    print_every = 1000
    lr = init_lr
    ks = None

    for s in range(max_steps):
        yhat = model(X)
        negll = 0.5 * (y - yhat).pow(2).mean()
        tv = eta_tv * model.tv_penalty(power=1)
        ridge = eta_ridge * model.shrink_penalty(power=1)
        loss = negll + tv + ridge

        if free_kernel:
            ks = model.kernel_smoothness()
            loss += eta_kernel_smoothness * ks

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                lr = max(param_group["lr"] * decay, 1e-5)
                param_group["lr"] = lr

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"negll: {float(negll):.6f}",
                f"tv: {float(tv):.6f}",
                f"ridge: {float(ridge):.6f}",
                f"total: {float(loss):.6f}",
                f"lr: {lr:.4f}",
            ]
            if free_kernel:
                msgs.append(f"ks: {float(ks):.6f}")
            print(", ".join(msgs))

    os.makedirs("outputs/spatial/{fsuffix}", exist_ok=True)
    torch.save(model.cpu(), f"outputs/spatial/{fsuffix}/weights.pt")
    print("done")

    gam = model.kernel.detach().cpu().norm(dim=-1)
    ix = list(reversed(range(0, nrows)))
    fig, ax = plt.subplots(1, 3)
    for p in range(units):
        ax[p].imshow(gam[ix, :, p].log().numpy())
        loc_p = locs[p]
        ax[p].scatter([loc_p[1]], [nrows - 1 - loc_p[0]], c="red")
        ax[p].set_title(f"Power plant {p}")
    plt.savefig(f"outputs/spatial/{fsuffix}/results_log.png")
    plt.close()
    fig, ax = plt.subplots(1, 3)
    for p in range(units):
        ax[p].imshow(gam[ix, :, p].numpy())
        loc_p = locs[p]
        ax[p].scatter([loc_p[1]], [nrows - 1 - loc_p[0]], c="red")
        ax[p].set_title(f"Power plant {p}")
    plt.savefig(f"outputs/spatial/{fsuffix}/results.png")
    plt.close()



if __name__ == "__main__":
    for what in ("no_seasonal", "seasonal", "double_seasonal"):
        for use_bias in (True, ):
            for free_kernel in (True, ):
                for use_log in (True, ):
                    for norm_x in (True, False):
                        for reg in (0.1, 10.0):
                            for non_linear in (False, ):
                                fsuffix = what
                                if use_bias:
                                    fsuffix += "_bias"
                                if use_log:
                                    fsuffix += "_log"
                                if free_kernel:
                                    fsuffix += "_free"
                                if norm_x:
                                    fsuffix += "_norm"
                                if reg > 0.1:
                                    fsuffix += "_hi_reg"
                                elif reg == 0.0:
                                    fsuffix += "_no_reg"
                                elif reg <= 0.1:
                                    fsuffix += "_lo_reg"
                                if non_linear:
                                    fsuffix += "_non_linear"

                                if free_kernel:
                                    init_lr = 0.5
                                else:
                                    init_lr = 5.0

                                print("Running:", fsuffix)
                                train(
                                    what=what,
                                    use_bias=use_bias,
                                    free_kernel=free_kernel,
                                    use_log=use_log,
                                    fsuffix=fsuffix,
                                    init_lr=init_lr,
                                    normalize_inputs=norm_x,
                                    reg=reg,
                                    non_linear=non_linear
                                )
