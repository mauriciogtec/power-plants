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
import seaborn as sn


def is_in(loc):
    xi, yi = loc
    min_lon = -80.55
    min_lat = 39.65
    max_lon = -75.25
    max_lat = 42.05
    return xi >= min_lon and xi <= max_lon and yi >= min_lat and yi <= max_lat


def to_intloc(loc):
    xi, yi = loc
    min_lon = -80.55
    min_lat = 39.65
    d = 0.01
    return int((xi - min_lon) / d), int((yi - min_lat) / d)


def dist2center(loc):
    xi, yi = loc
    min_lon = -80.55
    min_lat = 39.65
    max_lon = -75.25
    max_lat = 42.05
    xc = 0.5 * (min_lon + max_lon)
    yc = 0.5 * (min_lat + max_lat)
    return np.sqrt((xi - xc) ** 2 + (yi - yc) ** 2)


def train(
    use_bias: bool = True,
    free_kernel: bool = False,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
    reg=1.0,
    non_linear: bool = False,
    in_region_only: bool = False,
) -> None:
    os.makedirs(f"outputs/phil/{fsuffix}", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("model_dev_data/phil.npz")
    X_ = data["X"]
    y_ = data["y"]
    # sigs = data["sigs"]
    # xcoords = data["xcoords"]
    # ycoords = data["ycoords"]
    # fid = data["fid"]
    ym = data["ym"]
    locs = data["locs"]
    in_units = [is_in(locs[p]) for p in range(locs.shape[0])]
    dists = [dist2center(locs[p]) for p in range(locs.shape[0])]
    locs = [to_intloc(locs[p]) for p in range(locs.shape[0])]

    # miss = 1.0 - data["miss"]
    locs_x = [L[0] for L, x in zip(locs, in_units) if x]
    locs_y = [L[1] for L, x in zip(locs, in_units) if x]

    if in_region_only:
        X_ = X_[in_units]
        locs = [locs[i] for i, x in enumerate(in_units) if x]
        dists = [dists[i] for i, x in enumerate(in_units) if x]
        in_units = [True] * len(in_units)

    if use_log:
        y_ = np.log(y_ + 1)
        X_ = np.log(X_ + 1)

    residual = False
    if residual:
        y_ = y_ - y_.mean(-1, keepdims=True)

    scales = np.log(X_ + 1).std(-1, keepdims=True)
    sizes = [25 * d for x, d in zip(in_units, (X_ / X_.std()).std(-1)) if x]
    if normalize_inputs:
        if use_log:
            X_ /= X_.std()
        else:
            X_ /= scales
        y_ /= y_.std()

    # std_y_ = y_.std()
    # locs = data["locs"]

    # some warup plots
    df = pd.DataFrame(data=X_[:10].transpose(), index=ym)
    df.plot()
    plt.savefig(f"outputs/phil/{fsuffix}/0_pp_ts.png")
    plt.title("Power-plant pollution time series")
    plt.close()

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.savefig(f"outputs/phil/{fsuffix}/0_corr-mat.png")
    plt.title("Power-plant pollution correlation")
    plt.close()

    X = torch.FloatTensor(X_).to(dev)
    y = torch.FloatTensor(y_).to(dev)
    # miss = torch.FloatTensor(miss).to(dev)

    kernel_size = 4
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

    decay = 0.9
    decay_every = 200
    max_steps = 50_000
    min_lr = 0.00005

    optim = torch.optim.Adam(
        model.parameters(), init_lr, (0.9, 0.99), eps=1e-3
    )

    eta_ridge = reg
    eta_kernel_smoothness = 0.1
    eta_tv =  0.1  # 1e4  # power=1 -> 1.0,  power=2 -> 1e4
    print_every = 100
    ckpt_every = 100

    lr = init_lr
    ks = None
    # C = (1.0 - miss).sum()
    for s in range(max_steps):
        yhat = model(X)
        # negll = 0.5 * ((1.0 - miss) * (y - yhat)).pow(2).sum() / C
        negll = 0.5 * (y - yhat).pow(2).mean()
        tv = eta_tv * model.tv_penalty(power=1) * X_.shape[0]
        ridge = eta_ridge * model.shrink_penalty(power=1) * X_.shape[0]
        loss = negll + tv + ridge

        if free_kernel:
            ks = model.kernel_smoothness()
            loss += eta_kernel_smoothness * ks

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                lr = max(param_group["lr"] * decay, min_lr)
                param_group["lr"] = lr

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"negll: {float(negll):.6f}",
                f"tv: {float(tv):.6f}",
                f"shrink: {float(ridge):.6f}",
                f"total: {float(loss):.6f}",
                f"lr: {lr:.4f}",
            ]
            if free_kernel:
                msgs.append(f"ks: {float(ks):.6f}")
            print(", ".join(msgs))

        if s % ckpt_every == 0:
            torch.save(model.cpu(), f"outputs/phil/{fsuffix}/weights.pt")
            gam = model.kernel.detach().cpu().norm(dim=-1)
            # M = float(gam.max())
            # m = float(gam.min())
            # Ml = float(gam.log().max())
            # ml = float(gam.log().min())
            fig, ax = plt.subplots(3, 3)
            k = 0
            revix = list(reversed(range(0, nrows)))
            # revix = list(range(0, nrows))
            for p in range(units):
                loc_p = locs[p]
                if not in_units[p]:
                    continue
                ix = k // 3, k % 3
                ax[ix].imshow(gam[revix, :, p].log().numpy())
                ax[ix].scatter(
                    loc_p[0], nrows - 1 - loc_p[1], s=sizes[k], c="red",
                )
                ax[ix].set_title(f"Power plant {k}")
                k += 1
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_knorm_log.png")
            plt.close()

            _, ax = plt.subplots(3, 3)
            k = 0
            for p in range(units):
                loc_p = locs[p]
                if not in_units[p]:
                    continue
                ix = k // 3, k % 3
                ax[ix].imshow(gam[revix, :, p].numpy())
                ax[ix].scatter(
                    loc_p[0], nrows - 1 - loc_p[1], s=sizes[k], c="red"
                )
                ax[ix].set_title(f"Power plant {k}")
                k += 1
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_knorm.png")
            plt.close()

            fig, ax = plt.subplots(3, 3)
            k = 0
            lags = model.kernel.detach().cpu().mean((0, 1))
            for p in range(units):
                loc_p = locs[p]
                if not in_units[p]:
                    continue
                ix = k // 3, k % 3
                ax[ix].plot(lags[p, :].numpy())
                ax[ix].set_title(f"Lag coefs {k}")
                k += 1
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_lags.png")
            plt.close()

            _, ax = plt.subplots(4, 3)
            for k in range(4):
                yhatlast = (
                    yhat.detach().cpu().numpy()[:, :, -1 - k * kernel_size]
                )
                ylast = y_[:, :, -1 - k * kernel_size]
                ax[k, 0].imshow(ylast[revix])
                ax[k, 0].scatter(
                    locs_x,
                    [nrows - 1 - yi for yi in locs_y],
                    s=sizes,
                    c="red",
                    alpha=0.5,
                )
                ax[k, 0].set_title("Real")
                ax[k, 0].axis("off")
                ax[k, 1].imshow(yhatlast[revix])
                ax[k, 1].scatter(
                    locs_x,
                    [nrows - 1 - yi for yi in locs_y],
                    s=sizes,
                    c="red",
                    alpha=0.5,
                )
                ax[k, 1].set_title("Predicted")
                ax[k, 1].axis("off")
                ax[k, 2].imshow(np.abs(yhatlast - ylast)[revix])
                ax[k, 2].set_title("Diff")
                ax[k, 2].scatter(
                    locs_x,
                    [nrows - 1 - yi for yi in locs_y],
                    s=sizes,
                    c="red",
                    alpha=0.5,
                )
                ax[k, 2].axis("off")
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_pred.png")
            plt.close()

            fig, ax = plt.subplots(1, 2)
            influences = gam.mean((0, 1)).numpy()
            ax[0].hist(influences)
            ax[0].set_title("Influences histogram")
            ax[1].scatter(dists, influences)
            ax[1].set_title("Influence by distance")
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_hist.png")
            plt.close()

            alpha = model.alpha.detach().cpu().numpy()
            plt.imshow(alpha[revix])
            plt.title("Intercept")
            plt.savefig(f"outputs/phil/{fsuffix}/{s:05d}_intercept.png")
            plt.close()

            model = model.to(dev)


if __name__ == "__main__":
    for use_bias in (True,):
        for use_log in (True,):
            for norm_x in (True, False):
                for reg in (0.001, ):  # hi + power=2 -> 10
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

                        init_lr = 0.05

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
