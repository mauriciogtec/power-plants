import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from convgru import ConvGRUCell
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg') # no UI backend


class RNNSpatialRegression(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        hidden_size: int,
        nplants: int,
        nrows: int,
        ncols: int,
        activation = "tanh"
    ) -> None:
        super().__init__()
        self.gru = ConvGRUCell(
            input_size=nplants,
            hidden_size=nplants,
            kernel_size=kernel_size,
            groups=nplants
        )
        self.conv = nn.Conv2d(
            nrows * ncols * nplants,
            nrows * ncols,
            kernel_size=1,
            groups=nrows * ncols,
            bias=False
        )
        self.bias = nn.Parameter(torch.randn(nrows, ncols))
        self.states = None
        self.nplants = nplants
        self.nrows = nrows
        self.ncols = ncols

    def forward(self, inputs: Tensor) -> Tensor:
        """Input must be a tensor of plants x nrow x ncol"""
        # causal conv
        self.states = self.gru(inputs.unsqueeze(0), self.states)
        x = self.states.permute(0, 2, 3, 1).reshape(1, -1, 1, 1)
        out = self.conv(x)
        out = out.view(self.nrows, self.ncols)
        out = out + self.bias
        return out

    def tv_penalty(self, power: int = 2) -> Tensor:
        x = self.states
        dr = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().pow(power)
        dc = (x[:, :, -1:, :] - x[:, :, 1:, :]).abs().pow(power)
        loss = dr.mean() + dc.mean()

        return loss

    def shrink_penalty(self, power: int = 2) -> Tensor:
        loss = self.states.abs().pow(power).mean()

        return loss

    def reset_states(self) -> None:
        self.states = None


def train(
    what: str,
    use_log=False,
    normalize_inputs=False,
    fsuffix: str = "",
    init_lr: float = 1.0,
) -> None:
    os.makedirs("outputs/rnn/images", exist_ok=True)
    data = np.load(f"data/simulation/{what}.npz")
    X_ = data["power_plants"]
    y_ = data["states"]
    sigs = data["sigs"]
    locs = data["locs"]

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


    kernel_size = 12
    warmup_win = kernel_size - 1
    nplants, T = X.shape
    nrows, ncols, _ = y.shape

    # build X seq
    inputs = torch.zeros(T, nplants, nrows, ncols)
    for p, (xi, yi) in enumerate(locs):
        for t in range(T):
            inputs[t, p, xi, yi] = X[p, t]

    model_fun = RNNSpatialRegression

    model = model_fun(
        kernel_size=13,
        hidden_size=nplants,
        nplants=nplants,
        nrows=nrows,
        ncols=ncols,
    ).to(dev)


    decay = 0.5
    decay_every = 50
    max_steps = 500

    optim = torch.optim.Adam(
        model.parameters(), init_lr, (0.9, 0.99), eps=1e-3
    )

    eta_shrink = 0.001
    eta_tv = 0.001

    print_every = 10
    animate_every = 10

    for s in range(max_steps):
        yhat = torch.zeros(nrows, ncols, T)

        tv = 0.0
        shrink = 0.0
        for t in range(T):
            yhat[:, :, t] = model(inputs[t])
            tv = tv + model.tv_penalty(power=1)
            shrink = shrink + model.shrink_penalty(power=1)

        negll = 0.5 * (y[:,:,warmup_win:] - yhat[:,:,warmup_win:]).pow(2).mean()
        loss = negll + eta_tv * tv + eta_shrink * shrink

        optim.zero_grad()
        loss.backward()
        optim.step()
        model.reset_states()

        if (s + 1) % decay_every == 0:
            for param_group in optim.param_groups:
                param_group["lr"] = max(param_group["lr"] * decay, 1e-4)

        if s % print_every == 0:
            msgs = [
                f"step: {s}",
                f"negll: {float(negll):.6f}",
                f"tv: {float(tv):.6f}",
                f"shrink: {float(shrink):.6f}",
                f"total: {float(loss):.6f}",
            ]
            print(", ".join(msgs))

        if s % animate_every == 0:
            fig = plt.figure()
            ims = []
            for t in range(T):
                model(inputs[t])
                states = model.states[0].detach().numpy()[2]
                im = plt.imshow(states, animated=True)
                ims.append([im])
            ani = animation.ArtistAnimation(fig, ims)
            ani.save(f"./outputs/rnn/images/test_{s:03d}.gif")
            plt.close()
            model.reset_states()



    torch.save(model.cpu(), f"outputs/weights_{fsuffix}.pt")
    print("done")


if __name__ == "__main__":
    for use_log in (True, False):
        for norm_x in (False, True):
            for what in ("no_seasonal", "seasonal", "double_seasonal"):
                fsuffix = what
                if use_log:
                    fsuffix += "_log"
                if norm_x:
                    fsuffix += "_norm"

                init_lr = 0.5

                print("Running:", fsuffix)
                train(
                    what=what,
                    use_log=use_log,
                    fsuffix=fsuffix,
                    init_lr=init_lr,
                    normalize_inputs=norm_x,
                )
