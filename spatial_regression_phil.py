import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import os
import pickle as pkl
import matplotlib.pyplot as plt
from models import LaggedSpatialRegression
import pandas as pd


def isin(loc, xcoords, ycoords):
    xi, yi = loc
    return (
        xi >= xcoords[0]
        and xi <= xcoords[-1]
        and yi >= ycoords[0]
        and yi <= ycoords[-1]
    )


def train(
    use_bias: bool = True,
    free_kernel: bool = False,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    reg=1.0,
    non_linear: bool = False,
) -> None:
    os.makedirs(f"outputs/phil/{fsuffix}", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("model_dev_data/phil.npz")
    X_ = data["X"]
    y_ = data["y"]
    # sigs = data["sigs"]
    locs = data["locs"]
    xcoords = data["xcoords"]
    ycoords = data["ycoords"]
    in_units = [isin(locs[p], xcoords, ycoords) for p in locs.shape[0]]
    # miss = 1.0 - data["miss"]

    if use_log:
        y_ = np.log(y_ + 1)
        X_ = np.log(X_ + 1)

    if normalize_inputs:
        scale = X_.std(1)
        X_ /= np.expand_dims(scale, -1)

    # std_y_ = y_.std()
    # locs = data["locs"]

    X = torch.FloatTensor(X_).to(dev)
    y = torch.FloatTensor(y_).to(dev)
    # miss = torch.FloatTensor(miss).to(dev)

    kernel_size = 3
    units, t = X.shape
    y = y[:, :, (kernel_size - 1) :]
    # miss = miss[:, :, (kernel_size - 1) :]
    nrows, ncols, _ = y.shape

    model = LaggedSpatialRegression(
        units=units,
        kernel_size=kernel_size,
        nrows=nrows,
        ncols=ncols,
        use_bias=use_bias,
        non_linear=non_linear,
    ).to(dev)

    decay = 0.8
    decay_every = 1000
    max_steps = 50000

    optim = torch.optim.Adam(
        model.parameters(), init_lr, (0.9, 0.99), eps=1e-3
    )

    eta_ridge = 0.02 * reg
    eta_kernel_smoothness = 0.1
    eta_tv = 0.02 * reg

    print_every = 100
    ckpt_every = 100

    lr = init_lr
    ks = None
    # C = (1.0 - miss).sum()
    for s in range(max_steps):
        yhat = model(X)
        # negll = 0.5 * ((1.0 - miss) * (y - yhat)).pow(2).sum() / C
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

        if s % ckpt_every == 0:
            torch.save(model.cpu(), f"outputs/phil/{fsuffix}/weights.pt")

            gam = model.kernel.detach().cpu().norm(dim=-1)
            ix = list(reversed(range(0, nrows)))
            fig, ax = plt.subplots(1, 3)
            for p in range(units):
                loc_p = locs[p]
                if not isin(loc_p, xcoords, ycoords):
                    continue
                ax[p].imshow(gam[ix, :, p].log().numpy())
                ax[p].scatter([loc_p[1]], [nrows - 1 - loc_p[0]], c="red")
                ax[p].set_title(f"Power plant {p}")
            plt.savefig(f"outputs/phil/{fsuffix}/results_log.png")
            plt.close()

            fig, ax = plt.subplots(1, min)
            for p in range(units):
                loc_p = locs[p]
                if not in_units[p]:
                    continue
                ax[p].imshow(gam[ix, :, p].numpy())
                ax[p].scatter([loc_p[1]], [nrows - 1 - loc_p[0]], c="red")
                ax[p].set_title(f"Power plant {p}")
            plt.savefig(f"outputs/phil/{fsuffix}/results.png")
            plt.close()


if __name__ == "__main__":
    for use_bias in (True,):
        for use_log in (True,):
            for norm_x in (True, False):
                for reg in (0.1, 10.0):
                    for non_linear in (False,):
                        fsuffix = "base"
                        if use_bias:
                            fsuffix += "_bias"
                        if use_log:
                            fsuffix += "_log"
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

                        init_lr = 0.1

                        print("Running:", fsuffix)
                        train(
                            use_bias=use_bias,
                            use_log=use_log,
                            fsuffix=fsuffix,
                            init_lr=init_lr,
                            normalize_inputs=norm_x,
                            reg=reg,
                            non_linear=non_linear,
                        )

if __name__ == "__main__":
    train()
