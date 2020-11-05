import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


ACTIVATIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}


class DeepLaggedSpatialRegression(nn.Module):
    def __init__(
        self,
        kernel_sizes: list,
        plants: int,
        nrows: int,
        ncols: int,
        activation = "tanh"
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.ncols = ncols
        self.nrows = nrows
        self.plants = plants
        self.units = ncols * nrows * plants
        self.act = ACTIVATIONS[activation]
        self.num_layers = len(kernel_sizes)

        k0, k1 = self.kernel_sizes
        self.weights_0 = nn.Parameter(torch.randn(nrows, ncols, plants, k0))
        self.bias_0 = nn.Parameter(torch.randn(nrows, ncols, plants))
        self.weights_1 = nn.Parameter(torch.randn(nrows, ncols, plants, k1))
        self.bias_1 = nn.Parameter(torch.randn(nrows, ncols))


    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.unsqueeze(0)  # expand batch dim
        weights = self.weights_0.view(-1, 1, self.kernel_sizes[0])
        bias = self.bias_0.view(-1)
        x = F.pad(x, (self.kernel_sizes[0] - 1, 0))  # for causal conv
        x = F.conv1d(x, weights, bias, groups=self.plants)
        x = self.act(x)
        x = x.view(1, self.nrows * self.ncols * self.plants, -1)

        weights = self.weights_1.view(-1, self.plants, self.kernel_sizes[1])
        bias = self.bias_1.view(-1)
        x = F.pad(x, (self.kernel_sizes[1] - 1, 0))
        x = F.conv1d(x, weights, bias, groups=self.nrows * self.ncols)

        x = x.view(self.nrows, self.ncols, -1)

        return x

    def tv_penalty(self, power: int = 2) -> Tensor:
        loss = 0.0

        for w in (self.weights_0, self.weights_1):
            dr = torch.abs(w[:, :-1, :, :] - w[:, 1:, :, :]).pow(power)
            dc = torch.abs(w[-1:, :, :, :] - w[1:, :, :, :]).pow(power)
            loss = loss + dr.mean() + dc.mean()

        for b in (self.bias_0, ):
            dr = torch.abs(b[:, :-1, :] - b[:, 1:, :]).pow(power)
            dc = torch.abs(b[-1:, :] - b[1:, :, :]).pow(power)
            loss = loss + dr.mean() + dc.mean()        

        for b in (self.bias_1, ):
            dr = torch.abs(b[:, :-1] - b[:, 1:]).pow(power)
            dc = torch.abs(b[-1:, ] - b[1:, :]).pow(power)
            loss = loss + dr.mean() + dc.mean()

        return loss

    def kernel_smoothness(self, power: int = 2) -> Tensor:
        loss = 0.0
        for w in (self.weights_0, self.weights_1):
            dz = (w[:, :, :, :-1] - w[:, :, :, 1:]).abs().pow(power)
            loss = loss + dz.mean()
        return loss

    def shrink_penalty(self, power: int = 2) -> Tensor:
        loss = 0.0
        for w in (self.weights_0, self.weights_1):
            loss = loss + w.norm(dim=-1).pow(power).mean()
        return loss


def train(
    what: str,
    use_bias: bool,
    free_kernel: bool = False,
    layers: int = False,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    reg=1.0
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
    # locs = data["locs"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.FloatTensor(X_).to(dev)
    y = torch.FloatTensor(y_).to(dev)

    layers = 2
    kernel_sizes = [12, 3]
    units, t = X.shape
    y = y[:, :, (kernel_sizes[0] - 1) :]
    nrows, ncols, _ = y.shape

    model_fun = DeepLaggedSpatialRegression

    model = model_fun(
        kernel_sizes=kernel_sizes,
        plants=units,
        nrows=nrows,
        ncols=ncols
    ).to(dev)

    decay = 0.8
    decay_every = 1000
    max_steps = 50000

    optim = torch.optim.Adam(model.parameters(), init_lr, (0.9, 0.99), eps=1e-3)

    eta_ridge = 0.02 * reg
    eta_kernel_smoothness = 0.1
    eta_tv = 0.02 * reg

    print_every = 1000

    T = y.shape[-1]
    for s in range(max_steps):
        yhat = model(X)[:, :, -T:]

        negll = 0.5 * (y - yhat).pow(2).mean()
        tv = eta_tv * model.tv_penalty(power=1)
        ridge = eta_ridge * model.shrink_penalty(power=1)
        loss = negll + tv + ridge
        ks = model.kernel_smoothness()
        loss += eta_kernel_smoothness * ks

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                param_group["lr"] = max(param_group["lr"] * decay, 1e-5)

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

    torch.save(model.cpu(), f"outputs/deep/weights_{fsuffix}.pt")
    print("done")


if __name__ == "__main__":
    for what in ("no_seasonal", "seasonal", "double_seasonal"):
        for use_bias in (True, ):
            for free_kernel in (True, ):
                for use_log in (True, ):
                    for norm_x in (True, ):
                        for reg in (10.0, 0.01):

                            fsuffix = what
                            if use_bias:
                                fsuffix += "_bias"
                            if use_log:
                                fsuffix += "_log"
                            if free_kernel:
                                fsuffix += "_free"
                            if norm_x:
                                fsuffix += "_norm"
                            if reg > 1.0:
                                fsuffix += "_hireg"
                            elif reg == 0.0:
                                fsuffix += "_no_reg"
                            elif reg < 1.0:
                                fsuffix += "_lo_reg"

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
                                reg=reg
                            )

