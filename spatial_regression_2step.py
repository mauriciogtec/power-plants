import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


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
        nrows: int,
        ncols: int,
        use_bias: bool = True,
        mask: Tensor = None,
        spurious: Tensor = None
    ) -> None:
        super().__init__()
        self.kernel = nn.Parameter(
            torch.randn(nrows, ncols, units, kernel_size)
        )
        self.kernel_size = kernel_size
        self.units = units
        self.nrows = nrows
        self.mask = mask
        self.ncols = ncols
        self.use_bias = use_bias
        if use_bias:
            self.alpha = nn.Parameter(torch.randn(nrows, ncols))
        # self.spurious = spurious
        # if self.spurious is not None:
        #     self.spurious_wts = nn.Parameter(
        #         torch.randn(nrows, ncols)
        #     )
        #     self.spurious_bias = nn.Parameter(
        #         torch.randn(nrows, ncols)
        #     )
        #     self.spurious = spurious

    def forward(self, inputs: Tensor) -> Tensor:
        kernel = self.kernel
        if self.mask is not None:
            kernel = self.mask.unsqueeze(-1) * kernel
        kernel = kernel.view(-1, self.units, self.kernel_size)

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
        # if self.spurious is not None:
        #     W = self.spurious_wts.unsqueeze(-1)
        #     b = self.spurious_bias.unsqueeze(-1)
        #     out = out + W * self.spurious + b

        return out

    def tv_penalty(self, power: int = 2) -> Tensor:
        x = self.kernel
        dr = torch.abs(x[:, :-1, :, :] - x[:, 1:, :, :]).pow(power)
        dc = torch.abs(x[-1:, :, :, :] - x[1:, :, :, :]).pow(power)
        loss = dr.mean() + dc.mean()

        return loss

    def kernel_smoothness(self, power: int = 2) -> Tensor:
        x = self.kernel
        dz = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().pow(power)
        return dz.mean()

    def shrink_penalty(self, power: int = 2) -> Tensor:
        loss = self.kernel.norm(dim=-1).pow(power).mean()

        return loss


def train(
    what: str,
    use_bias: bool,
    free_kernel: bool = False,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
) -> None:
    data = np.load(f"data/simulation/{what}.npz")
    X_ = data["power_plants"]
    y_ = data["states"]
    sigs = data["sigs"]

    if normalize_inputs:
        X_ /= np.expand_dims(sigs, -1)

    if use_log:
        y_ = np.log(y_ + 1e-6)
        X_ = np.log(X_ + 1e-6)

    # std_y_ = y_.std()
    locs = data["locs"]

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

    decay = 0.5
    decay_every = 5000
    max_steps_spurious = 20000
    max_steps = 50000

    print_every = 1000

    # stage 1 builds a predictive model using known spurious features
    # mask is one of spurious
    mask = np.zeros((nrows, ncols, units), dtype=int)
    for i in range(nrows):
        for j in range(ncols):
            for p in range(units):
                x, y = locs[p]
                if (x > i) or (y > j):
                    mask[i, j, p] = 1  # spurious
    mask = torch.FloatTensor(mask).cuda()

    model = model_fun(
        units=units,
        kernel_size=kernel_size,
        nrows=nrows,
        ncols=ncols,
        use_bias=use_bias,
        mask=mask
    ).cuda()

    optim = torch.optim.Adam(model.parameters(), init_lr, (0.9, 0.99), eps=1e-3)

    eta_ridge = 0.01
    eta_kernel_smoothness = 0.01
    eta_tv = 0.1

    for s in range(max_steps_spurious):
        spurious_yhat = model(X)
        loss = 0.5 * (y - spurious_yhat).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                param_group["lr"] = max(param_group["lr"] * decay, 1e-4)

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"total: {float(loss):.6f}",
            ]
            print(", ".join(msgs))

    torch.save(model.cpu(), f"outputs/weights_{fsuffix}_spur.pt")
    print("done training spurious reg")

    model = model_fun(
        units=units,
        kernel_size=kernel_size,
        nrows=nrows,
        ncols=ncols,
        use_bias=use_bias,
    ).cuda()

    U = spurious_yhat.detach()

    optim = torch.optim.Adam(model.parameters(), init_lr, (0.9, 0.99), eps=1e-3)
    eta_ridge = 0.1
    eta_kernel_smoothness = 0.1
    eta_tv = 0.01

    for param_group in optim.param_groups:
        param_group["lr"] = init_lr

    for s in range(max_steps):
        yhat = model(X)
        negll = 0.5 * ((y - U) - yhat).pow(2).mean()
        tv = eta_tv * model.tv_penalty(power=1)
        ridge = eta_ridge * model.shrink_penalty(power=1)
        loss = negll + tv + ridge

        if free_kernel:
            ks = model.kernel_smoothness()
            loss += eta_kernel_smoothness * ks

        loss = negll + tv + ridge

        if free_kernel:
            ks = model.kernel_smoothness()
            loss += eta_kernel_smoothness * ks

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                param_group["lr"] = max(param_group["lr"] * decay, 1e-4)

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"negll: {float(negll):.6f}",
                f"tv: {float(tv):.6f}",
                f"ridge: {float(ridge):.6f}",
                f"total: {float(loss):.6f}",
            ]
            if free_kernel:
                msgs.append(f"ks: {float(ks):.6f}")
            print(", ".join(msgs))

    torch.save(model.cpu(), f"outputs/weights_{fsuffix}_2step.pt")
    print("done training spurious reg")


if __name__ == "__main__":
    for what in ("seasonal", ):
        for use_bias in (True, False):
            for free_kernel in (True, False):
                for use_log in (True, False):
                    for norm_x in (False, True):

                        fsuffix = what
                        if use_bias:
                            fsuffix += "_bias"
                        if use_log:
                            fsuffix += "_log"
                        if free_kernel:
                            fsuffix += "_free"
                        if norm_x:
                            fsuffix += "_norm"

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
                            normalize_inputs=norm_x
                        )

