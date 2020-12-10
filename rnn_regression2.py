import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from convgru2 import ConvGRUCell
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional


matplotlib.use("Agg")  # no UI backend


class RNNSpatialRegression(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        nplants: int,
        nrows: int,
        ncols: int,
        reset_gate: bool = True,
        update_gate: bool = True,
        renormalize_inputs: bool = True,
        min_update: float = 0.0,
        min_reset: float = 0.0,
        rnn_act: Optional[str] = None,
        additive: bool = False,
    ) -> None:
        super().__init__()
        # self.gru = ConvGRUCell(
        #     input_size=nplants,
        #     hidden_size=nplants,
        #     kernel_size=kernel_size,
        #     groups=nplants,
        # )
        self.gru = ConvGRUCell(
            input_size=1,
            hidden_size=1,
            kernel_size=kernel_size,
            groups=1,
            reset_gate=reset_gate,
            update_gate=update_gate,
            out_bias=(not renormalize_inputs),
            out_act=rnn_act,
            min_update=min_update,
            min_reset=min_reset,
        )
        if not additive:
            self.final_layer = True
            self.add_weights = nn.Parameter(
                0.02 * torch.randn(nplants, nrows, ncols)
            )
        else:
            self.final_layer = None
        self.bias = nn.Parameter(torch.zeros(nrows, ncols))
        self.nplants = nplants
        self.nrows = nrows
        self.ncols = ncols
        self.rnn_act = rnn_act

        if renormalize_inputs:
            self.renormalize_inputs = True
            self.gam = nn.Parameter(torch.ones(nplants))
            # self.beta = nn.Parameter(torch.zeros(nplants))
        else:
            self.renormalize_inputs = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Input must be a tensor of time x plants x nrow x ncol"""
        # causal conv
        T = inputs.shape[0]
        dev = inputs.device
        out_y = torch.empty((T, self.nrows, self.ncols))
        out_y = out_y.to(dev)

        out_states = torch.empty((T, self.nplants, self.nrows, self.ncols))
        out_states = out_states.to(dev)

        # inefficient batchnorm
        if self.renormalize_inputs:
            gam = self.gam.view(1, -1, 1, 1)
            # beta = self.beta.view(-1, 1, 1, 1)
            inputs = gam * inputs  # + beta

        states = torch.zeros_like(inputs[0].unsqueeze(1))
        for t in range(T):
            states = self.gru(inputs[t].unsqueeze(1), states)
            x = states.squeeze(1)

            if self.final_layer:
                # x = states.permute(1, 2, 3, 0).contiguous().view(1, -1, 1, 1)
                # x = self.final_layer(x)
                # x = x.view(self.nrows, self.ncols)
                if not self.rnn_act:
                    x = torch.tanh(x)
                x = x * self.add_weights

            out_y[t] = x.sum(0) + self.bias
            out_states[t] = x

        return out_y, out_states


def tv_penalty(x, power: int = 2) -> Tensor:
    dims = list(range(len(x.shape)))
    dims[:2], dims[-2:] = dims[-2:], dims[:2]
    x = x.permute(*dims)
    dr = (x[:, :-1] - x[:, 1:]).abs().pow(power)
    dc = (x[-1:, :] - x[1:, :]).abs().pow(power)
    loss = dr.mean() + dc.mean()
    return loss


def shrink_penalty(x, power: int = 2) -> Tensor:
    loss = x.abs().pow(power).mean()
    return loss


def train(
    what: str,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    eta_shrink: float = 1.0,
) -> None:
    os.makedirs("outputs/rnn/images", exist_ok=True)
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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.FloatTensor(X_).to(dev)
    y = torch.FloatTensor(y_).permute(2, 0, 1).to(dev)

    kernel_size = 5
    warmup_win = 0  # kernel_size - 1
    nplants, T = X.shape
    _, nrows, ncols = y.shape

    # build X seq
    inputs = torch.zeros(T, nplants, nrows, ncols)
    for p, (xi, yi) in enumerate(locs):
        for t in range(T):
            inputs[t, p, xi, yi] = X[p, t]
    inputs = inputs.to(dev)

    model = RNNSpatialRegression(
        kernel_size=kernel_size,
        nplants=nplants,
        nrows=nrows,
        ncols=ncols,
        update_gate=False,
        reset_gate=False,
        # min_update=0.5,
        min_reset=0.8,
        renormalize_inputs=True,
        additive=False,
        # rnn_act="tanh",
    ).to(dev)

    decay = 0.5
    decay_every = 5000
    max_steps = 15_000

    init_lr = 0.01
    burnin = 1000
    burnin_decay = 100

    optim = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=(0.9, 0.99),
        eps=1e-3,
        weight_decay=eta_shrink,
    )
    # optim = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-5)

    eta_tv = 0.01
    eta_ts = 1.0
    eta_mass = 100.0

    print_every = 50
    animate_every = 500

    for s in range(max_steps):
        yhat, states = model(inputs)
        delta = y - yhat
        negll = 0.5 * delta[warmup_win:, :, :].pow(2).mean()
        tv = 0.0
        shrink = 0.0
        time_smooth = 0.0

        # excess mass penalty
        available_mass = states[:-1].sum((2, 3)) + inputs[1:].sum((2, 3))
        mass_diff = states[1:].sum((2, 3)) - available_mass
        loss_mass = torch.relu(mass_diff).pow(2).mean()

        # tv = tv_penalty(model.bias, power=2)
        # shrink_bias = shrink_penalty(model.bias, power=1)
        # shrink = shrink_penalty(states, power=1)
        time_smooth = (states[1:] - states[:-1]).pow(2).mean()
        loss = (
            negll
            + eta_tv * tv
            + eta_shrink * shrink
            + eta_ts * time_smooth
            + eta_mass * loss_mass
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optim.step()
        model.gru.out_weights.data.clamp_(0)

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                lr = max(param_group["lr"] * decay, 1e-4)
                param_group["lr"] = lr

        if s % burnin_decay == 0 and s < burnin + 1:
            for param_group in optim.param_groups:
                lr = max(init_lr * min(1.0, (s + 1) / burnin), 1e-5)
                param_group["lr"] = lr

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"negll: {float(negll):.6f}",
                f"tv: {float(tv):.6f}",
                f"shrink: {float(shrink):.6f}",
                f"total: {float(loss):.6f}",
                f"ts: {float(time_smooth):.6f}",
                f"mass: {float(loss_mass):.6f}",
                f"lr: {float(lr):.4f}",
            ]
            print(", ".join(msgs))

        os.makedirs(f"./outputs/rnn/images/{fsuffix}/", exist_ok=True)
        if s % animate_every == 0:
            for p in range(3):
                fig = plt.figure()
                ims = []
                for t in range(T):
                    sp = states[t, p].detach().cpu().numpy()
                    im = plt.imshow(sp, animated=True)
                    ims.append([im])
                ani = animation.ArtistAnimation(fig, ims)
                ani.save(
                    f"./outputs/rnn/images/{fsuffix}/test_{s:03d}_{p}.gif"
                )
                plt.close()
            fig = plt.figure()
            ims = []
            for t in range(T):
                d = delta[t].detach().cpu().numpy()
                im = plt.imshow(d, animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims)
            ani.save(f"./outputs/rnn/images/{fsuffix}/test_{s:03d}_delta.gif")
            plt.close()

    torch.save(model.cpu(), f"outputs/weights_{fsuffix}.pt")
    print("done")


if __name__ == "__main__":
    i = 0
    for use_log in (False, True):
        for norm_x in (False, True):
            for eta_shrink in (0.0001, 0.0):
                for what in ("no_seasonal", "seasonal", "double_seasonal"):
                    fsuffix = what
                    if use_log:
                        fsuffix += "_log"
                    if norm_x:
                        fsuffix += "_norm"
                    if eta_shrink > 0.0:
                        fsuffix += f"_shrink{eta_shrink}"

                    init_lr = 0.005

                    print("Running:", fsuffix)
                    train(
                        what=what,
                        use_log=use_log,
                        fsuffix=fsuffix,
                        init_lr=init_lr,
                        normalize_inputs=norm_x,
                        eta_shrink=eta_shrink,
                    )
