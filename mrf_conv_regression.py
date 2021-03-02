# %%
import numpy as np
import torch
from torch import FloatTensor, LongTensor, nn, Tensor, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle5 as pkl
from utils import huber
import wandb
import pandas as pd
import geopandas as gpd
from typing import Optional


# %%
hyperparameter_defaults = dict(
    autoreg=False,
    normalize_by_plant=False,
    clamp_weights=False,
    tv=0.12,
    shrink=0.27,
    huber_k=-2.99,
    gravity=2.0
)

wandb.init(project="power-plants", config=hyperparameter_defaults)

print(f"Running with config {wandb.config}")
config = wandb.config

tv = 10 ** config.tv
shrink = 10 ** config.shrink
huber_k = 10 ** config.huber_k
gravity = 10 ** config.gravity


# %% read spatial graph

with open("model_dev_data/graph_training_data.pkl", "rb") as io:
    data = pkl.load(io)

data.keys()

coords = data["zip_coords"]
coords = gpd.GeoDataFrame(
    coords,
    geometry=gpd.points_from_xy(coords.lon, coords.lat)
)
pp_coords = data['locs']
pp_coords = gpd.GeoDataFrame(
    pp_coords,
    geometry=gpd.points_from_xy(pp_coords.lon, pp_coords.lat)
)


# %%


class PlantsLinearModel(nn.Module):
    def __init__(
        self,
        num_units,
        num_plants,
        init_positive=False
    ):
        super().__init__()
        self.N = num_units
        self.P = num_plants
        self.fx = nn.Linear(self.P, self.N, bias=False)

        # initialization for fx
        if init_positive:
            stdv = 1. / np.sqrt(self.P)
            self.fx.weight.data.uniform_(0.01 * stdv, stdv)

    def forward(self, X: Tensor):
        pred = self.fx(X)
        return pred

    def tv_loss(self, src: LongTensor, tgt: LongTensor, edgew: FloatTensor):
        W = self.fx.weight  # N X P matrix
        wdelta2 = (W[src] - W[tgt]).pow(2).mean(1)
        loss = (wdelta2 * edgew).sum()
        return loss

    def shrink_loss(self, huber_k: float = 1.0):
        W = self.fx.weight
        loss = huber(W, huber_k).mean()
        return loss

    def log_barrier(self):
        return - 1e-6 * torch.log(self.fx.weight).sum()

    def clamp_weights(self):
        self.fx.weight.data.clamp_(1e-4)

    def gravity_penalty(self, D: Tensor, huber_k: float = 1.0):
        # D is N X P
        W = self.fx.weight  # N x P
        dist_penalty = huber(W * D, huber_k).mean()
        return dist_penalty


class CovarsConvEmbedding(nn.Module):
    def __init__(
        self,
        embedding,
        num_covars=7,
        filters_first=16,
        filters_factor=2,
        dilation=2
    ):
        super().__init__()
        self.embedding = embedding.unsqueeze(0)

        def downsaple_layer(ins, outs):
            layer = nn.Sequential(
                nn.Conv2d(
                    ins, outs, 4, padding=2, bias=False, dilation=dilation
                ),
                nn.BatchNorm2d(outs),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2 * dilation, padding=0)
            )
            return layer

        def upsample_layer(ins, outs):
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    ins, outs, kernel_size=4, stride=4, padding=0, bias=False
                ),
                nn.BatchNorm2d(outs),
                nn.LeakyReLU()
            )
            return layer

        # convolutional model for the features
        H = filters_factor
        D = filters_first
        self.down1 = downsaple_layer(num_covars, D)
        self.down2 = downsaple_layer(D, H * D)
        self.down3 = downsaple_layer(H * D, H**2 * D)
        self.down4 = downsaple_layer(H**2 * D, H**3 * D)
        self.up1 = upsample_layer(H**3 * D, H**2 * D)
        self.up2 = upsample_layer(2 * H**2 * D, H * D)
        self.up3 = upsample_layer(2 * H * D, D)
        self.up4 = upsample_layer(2 * D, num_covars)
        self.final = nn.Conv2d(2 * num_covars, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, embed: bool = True):
        d1 = self.down1(inputs)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        x = torch.cat([self.up1(d4), d3], 1)
        x = torch.cat([self.up2(x), d2], 1)
        x = torch.cat([self.up3(x), d1], 1)
        x = self.up4(x)
        x = torch.cat([x, inputs], 1)
        x = self.final(x)
        if embed:
            # from grid to vector
            x = x.view(x.shape[0], -1)
            # take embedding
            x = torch.gather(x, 1, self.embedding)
        return x




# %%


nodes = data["nodes"]
nodemap = {x: i for i, x in enumerate(nodes)}
adj = data["adj"]
src = [nodemap[x] for x in adj.src_lab]
tgt = [nodemap[x] for x in adj.tgt_lab]
edgew = 1.0 / adj.dist.values
edgew /= edgew.std()


# %% load HyADS for validation
hyads_months = ("01", "06", "12")
W0s = []
for m in hyads_months:
    f = f"model_dev_data/hyads/2015/{m}.csv"
    fids = [str(x) for x in data['fid']]
    hyads = pd.read_csv(f)
    hyads.ZIP = [f"{x:05d}" for x in hyads.ZIP]
    hyads = hyads.set_index("ZIP")[fids]
    hyads = hyads.loc[nodes].values
    W0s.append(hyads)


# %%
dev = "cuda" if torch.cuda.is_available else "cpu"

autoreg = config.autoreg  #  True
normalize = True
normalize_by_plant = config.normalize_by_plant  # False


X = data["X"]
Y = data["Y"]
C = data["covars_rast"]
S = data["so4_rast"]
E = data["rast_emb"]
D = data['zip_to_pp_dists']

# %%

# keep in float64
for W0 in W0s:
    W0 **= 0.2
    W0 -= W0.mean()
    W0 /= W0.std()
    # W0 /= W0.std()

X = X ** 0.2
Y = Y ** 0.2

if normalize:
    if normalize_by_plant:
        # X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
        X /= X.std(0, keepdims=True)
    else:
        # X = (X - X.mean()) / X.std()
        X /= X.std()
    # Y = (Y - Y.mean()) / Y.std()
    Y /= Y.std()

if autoreg:
    Ylag = Y[0, ...]
    Y = Y[1:, ...]
    X = X[1:, ...]
    C = C[1:, ...]
    C = np.concatenate([C, np.expand_dims(Y, -1)], -1)

X = torch.FloatTensor(X).to(dev)
Y = torch.FloatTensor(Y).to(dev)
S = torch.FloatTensor(S).to(dev)
C = torch.FloatTensor(C).to(dev)
E = torch.LongTensor(E).to(dev) 
D = torch.FloatTensor(D).to(dev)


# %%
src = torch.LongTensor(src).to(dev)
tgt = torch.LongTensor(tgt).to(dev)
edgew = torch.FloatTensor(edgew).to(dev)


# %%
T, L, nr, nc = C.shape
T, P = X.shape
T, N = Y.shape

mod = PlantsLinearModel(
    num_units=N,
    num_plants=P,
    init_positive=config.clamp_weights
).to(dev)

mod_c = CovarsConvEmbedding(
    embedding=E,
    num_covars=L,
    filters_first=16,
    filters_factor=2,
    dilation=2
).to(dev)

opt = optim.Adam(
    list(mod.parameters()) + list(mod_c.parameters()),
    weight_decay=1e-4,
    lr=0.1,
    betas=(0.9, 0.99)
)
scheduler = ReduceLROnPlateau(
    opt, 'min', factor=0.5, patience=200, verbose=True
)


# %%
epochs = 7_500
print_every = 100
val_every = 1000
batch_size = 8


# %%

# %%
clamp_weights = config.clamp_weights

Yhat = None  # declare as global scope
effects = [None for _ in hyads_months]

for e in range(epochs):
    # batch = torch.randint(0, C.shape[0], (batch_size, ), device=dev)
    effs = mod(X)
    confs = mod_c(C)
    Yhat = confs + effs
    ll_loss = huber(Y - Yhat, k=1.0).mean()
    tv_loss = mod.tv_loss(src, tgt, edgew)
    shrink_loss = mod.shrink_loss(huber_k)
    gravity_loss = mod.gravity_penalty(D, huber_k)

    if clamp_weights:
        barr_loss = mod.log_barrier()  # optional for pos weights
    else:
        barr_loss = 0.0

    loss = (
        ll_loss
        + tv * tv_loss
        + shrink * shrink_loss
        + gravity * gravity_loss
        + 1e-6 * barr_loss
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    if clamp_weights:
        mod.clamp_weights()  # optional for pos weights

    if e == 0 or (e + 1) % print_every == 0:
        metrics = dict(
            ll=float(ll_loss),
            tv=float(tv_loss),
            shrink=float(shrink_loss),
            gravity=float(gravity_loss),
            barr=float(barr_loss),
            step=e
        )
        print(', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        wandb.log(metrics, step=e)

    if e == 0 or (e + 1) % val_every == 0:
        val_loss = 0.0
        for j, m in enumerate(hyads_months):
            t = (2005 - 2000) * 12 + int(m) - 1 - int(autoreg)

            Xt = X[t, ...].unsqueeze(0).cpu().numpy().astype(np.float64)
            weight = mod.fx.weight.detach().cpu().numpy().astype(np.float64)

            Wd = Xt * weight  # effect of each power plant
            Wd = (Wd - np.mean(Wd)) / (Wd.std() + 1e-32)
            # Wd /= (Wd.norm() + 1e-32)
            effects[j] = Wd

            # correlation is not reliable due to rounding errors
            val_loss -= (Wd * W0s[j]).mean() / len(hyads_months)

        print(f"val_loss: {val_loss:.4f}")
        wandb.log({'val_loss': val_loss}, step=e)

    scheduler.step(loss)
    scheduler_c.step(loss)

## %%
W = mod.fx.weight.detach().cpu().numpy()

# %%  random histograms of effects
cols = ["blue", "orange", "purple"]

rank = W.mean(0).flatten().argsort()
for i, ix in enumerate([0, 4 * P // 5, P - 1]):
    plt.hist(W[:, rank[ix]], color=cols[i], alpha=0.2)

plt.title("Effects of SOME plants")
wandb.log({"hist_ex": wandb.Image(plt)})
plt.close()

# %%  all histograms

plt.hist(W.flatten())
plt.title("Effects of ALL plants")
wandb.log({"hist_all": wandb.Image(plt)})
plt.close()

# %%
nplots = 10
Wd = effects[0]
size = Wd.mean(0).argsort()
for i in range(nplots):
    ix = size[-(i + 1)]
    # plot results
    coords['val'] = Wd[:, ix]
    nr, nc = i // 3, i % 3
    fig, ax = plt.subplots()
    coords.plot(
        column='val',
        markersize=1.0,
        alpha=0.5,
        cmap='magma',
        ax=ax
    )
    pp_coords.iloc[ix:(ix + 1)].plot(
        ax=ax,
        c="green",
        markersize=100
    )
    plt.axis("off")
    plt.title(f"Effects on 2005/01 of power plant {ix}")
    wandb.log({"effects": wandb.Image(plt)})
    plt.close()

    # now for hyads
    coords['val'] = W0s[0][:, ix]
    fig, ax = plt.subplots()
    coords.plot(
        column='val',
        markersize=1.0,
        alpha=0.5,
        cmap='magma',
        ax=ax
    )
    pp_coords.iloc[ix:(ix + 1)].plot(
        ax=ax,
        c="green",
        markersize=100
    )
    plt.axis("off")
    plt.title(f"HyADS on 2005/01 of power plant {ix}")
    wandb.log({"hyads 2005/01": wandb.Image(plt)})
    plt.close()




# %% comparison y vs yhat

t = 0
Yhat_ = Yhat[t].detach().cpu().numpy().flatten()
Y_ = Y[t].cpu().numpy().flatten()

plt.scatter(Y_, Yhat_)
plt.title("Observed vs Fit")
plt.xlabel("Obs")
plt.ylabel("Fit")
wandb.log({"fit": wandb.Image(plt)})
plt.close()

# %%
