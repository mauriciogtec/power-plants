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


class Seasonal(nn.Module):
    def __init__(
        self,
        length: int,
        kernel_size: int,
        nrows: int,
        ncols: int,
        additive: bool = False,  # otherwise multiplicative
    ) -> None:
        super().__init__()
        self.length = length
        self.kernel_size = kernel_size
        self.additive = additive
        self.w = nn.Parameter(torch.zeros(nrows, ncols, kernel_size))
        self.alpha = nn.Parameter(torch.zeros(nrows, ncols))

    def forward(self) -> Tensor:
        reps = self.length // self.kernel_size
        resid = self.length % self.kernel_size
        if self.additive:
            x = self.alpha.unsqueeze(-1) + self.w
        else:
            x = self.alpha.unsqueeze(-1) * (1.0 + self.w)
        x_tiled = x.repeat((1, 1, reps))
        x_resid = x[:, :, :resid]
        x = torch.cat([x_tiled, x_resid], -1)
        return x

    def spatial_penalty(self, power: int = 2) -> list:
        losses = []
        for x in (self.w,):
            dr = (x[:, :-1] - x[:, 1:]).abs().pow(power)
            dc = (x[:-1, :] - x[1:, :]).abs().pow(power)
            losses.append(dr.mean() + dc.mean())
        return losses[0]

    def time_penalty(self, power: int = 2) -> list:
        losses = []
        for x in (self.w,):
            dt = (x[:, :, :-1] - x[:, :, 1:]).abs().pow(power)
            losses.append(dt.mean())
        return losses[0]

    def shrink_penalty(self, power: int = 2) -> list:
        losses = []
        x = self.w
        losses.append(self.w.abs().pow(power).mean())
        return losses[0]


def train(
    what: str,
    use_log=False,
    # normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    # eta_shrink: float = 1.0,
) -> None:
    os.makedirs("outputs/rnn/images", exist_ok=True)
    data = np.load(f"data/simulation/{what}.npz")
    X_ = data["power_plants"]
    y_ = data["states"]
    # sigs = data["sigs"]
    locs = data["locs"]

    if use_log:
        X_ = np.log(X_ + 1)
        y_ = np.log(y_ + 1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.FloatTensor(X_).to(dev)
    y = torch.FloatTensor(y_).to(dev)

    nplants, T = X.shape
    nrows, ncols, ntime = y.shape

    kernel_size = 12
    warmup_win = 0

    y = y[:, :, warmup_win:]
    N = ntime - warmup_win

    model = Seasonal(
        length=N,
        kernel_size=kernel_size,
        nrows=nrows,
        ncols=ncols,
        additive=use_log,
    ).to(dev)

    decay = 0.9
    decay_every = 5000
    max_steps = 5_000

    init_lr = 0.001
    burnin = 1
    burnin_decay = 100

    optim = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=(0.9, 0.99),
        eps=1e-3,
        # weight_decay=eta_shrink,
    )
    # optim = torch.optim.SGD(model.parameters(), lr, weight_decay=1e-5)

    print_every = 500
    animate_every = 5000
    lr = init_lr

    for s in range(max_steps + 1):
        yhat = model()
        negll = (y - yhat).pow(2).mean()
        sreg = model.spatial_penalty(power=1)
        treg = model.time_penalty(power=2)
        shrink = model.shrink_penalty(power=2)
        loss = negll + 5.0 * sreg + 0.1 * treg + 0.01 * shrink

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optim.step()

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
                f"spatial: {float(sreg):.6f}",
                f"time: {float(treg):.6f}",
                f"shrink: {float(shrink):.6f}",
                f"lr: {float(lr):.4f}",
            ]
            print(", ".join(msgs))

        os.makedirs(f"./outputs/seasonality/{fsuffix}/images/", exist_ok=True)
        if s % animate_every == 0:
            pars = {
                "w": model.w.detach().cpu().numpy(),
            }
            for name, x in pars.items():
                fig = plt.figure()
                ims = []
                for t in range(kernel_size):
                    im = plt.imshow(x[:, :, t], animated=True)
                    ims.append([im])
                ani = animation.ArtistAnimation(fig, ims)
                ani.save(
                    f"./outputs/seasonality/{fsuffix}/images/anim_{s:03d}_{name}.gif"
                )
                plt.close()
            fig, ax = plt.subplots(1, 3, figsize=(16, 8))
            for p in range(3):
                px, py = locs[p]
                w = model.w[px, py, :].detach().cpu().numpy()
                ax[p].plot(w)
                ax[p].set_title(f"Power plant {p} term")
            plt.savefig(
                f"./outputs/seasonality/{fsuffix}/images/locs_{s:03d}.png"
            )
            plt.close()

    torch.save(model.cpu(), f"outputs/seasonality/{fsuffix}/weights.pt")
    np.save(
        f"outputs/seasonality/{fsuffix}/season.npy",
        model().detach().cpu().numpy(),
    )
    print("done")


if __name__ == "__main__":
    i = 0
    for use_log in (False, True):
        # for norm_x in (False, True):
        for what in ("seasonal", "double_seasonal"):
            fsuffix = what
            if use_log:
                fsuffix += "_log"
            # if norm_x:
            #     fsuffix += "_norm"

            init_lr = 0.005
            print("Running:", fsuffix)
            train(
                what=what,
                use_log=use_log,
                fsuffix=fsuffix,
                init_lr=init_lr,
                # normalize_inputs=norm_x,
                # eta_shrink=eta_shrink,
            )
