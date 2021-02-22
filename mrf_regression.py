# %%
import numpy as np
import torch
from torch import FloatTensor, LongTensor, nn, Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle as pkl
from utils import huber
import wandb
import pandas as pd
import geopandas as gpd


# %%
hyperparameter_defaults = dict(
    autoreg=True,
    normalize_by_plant=True,
    clamp_weights=True,
    tv=2.0,
    shrink=2.0,
    huber_k=-2.0,
    gravity=0.1 
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


class Model(nn.Module):
    def __init__(
        self,
        num_units,
        num_covars,
        num_plants,
        hidden_units: int = 16,
    ):
        super().__init__()
        self.N = num_units
        self.P = num_plants
        self.L = num_covars
        self.H = hidden_units

        self.fx = nn.Linear(self.P, self.N)
        self.fc = nn.Sequential(
            nn.Linear(self.L, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, 1, bias=False)
            # nn.Linear(self.L, 1)
        )

        # initialization for fx
        stdv = 1. / np.sqrt(self.P)
        self.fx.weight.data.uniform_(0.01 * stdv, stdv)


    def forward(self, X: Tensor, C: Tensor):
        xout = self.fx(X)
        T, *_ = C.shape
        cout = self.fc(C.view(-1, self.L))
        cout = cout.view(T, self.N)
        pred = xout + cout
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

    def gravity_penalty(self, D: Tensor):
        # D is N X P
        W = self.fx.weight  # N x P
        dist_penalty = huber(W * D).mean()
        return dist_penalty



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
C = data["C"]
D = data['zip_to_pp_dists']

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
C = torch.FloatTensor(C).to(dev) 
D = torch.FloatTensor(D).to(dev)


# %%
src = torch.LongTensor(src).to(dev)
tgt = torch.LongTensor(tgt).to(dev)
edgew = torch.FloatTensor(edgew).to(dev)


# %%
T, N, L = C.shape
T, P = X.shape
T, N = Y.shape

mod = Model(
    num_units=N,
    num_covars=L,
    num_plants=P,
    hidden_units=16,
).to(dev)

opt = optim.Adam(
    mod.parameters(),
    lr=0.1,
    betas=(0.9, 0.99)
)
scheduler = ReduceLROnPlateau(
    opt, 'min', factor=0.5, patience=200, verbose=True
)


# %%
epochs = 15_000
print_every = 100
val_every = 1000

# %%

# %%
clamp_weights = config.clamp_weights

Yhat = None  # declare as global scope
for e in range(epochs):
    Yhat = mod(X, C)
    ll_loss = huber(Y - Yhat, k=1.0).mean()
    tv_loss = mod.tv_loss(src, tgt, edgew)
    shrink_loss = mod.shrink_loss(huber_k)
    gravity_loss = mod.gravity_penalty(D)

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

    effects = []
    if e == 0 or (e + 1) % val_every == 0:
        val_loss = 0.0
        for j, m in enumerate(hyads_months):
            t = (2005 - 2000) * 12 + int(m) - 1 - int(autoreg)

            Xt = X[t, ...].unsqueeze(0).cpu().numpy().astype(np.float64)
            weight = mod.fx.weight.detach().cpu().numpy().astype(np.float64)

            Wd = Xt * weight  # effect of each power plant
            Wd = (Wd - np.mean(Wd)) / (Wd.std() + 1e-32)
            # Wd /= (Wd.norm() + 1e-32)
            effects.append(Wd)

            # correlation is not reliable due to rounding errors
            val_loss -= (Wd * W0s[j]).mean() / len(hyads_months)

        print(f"val_loss: {val_loss:.4f}")
        wandb.log({'val_loss': val_loss}, step=e)

    scheduler.step(loss)

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
        markersize=10
    )
    plt.axis("off")
    plt.title(f"Effects on 2005/01 of most impactful plants #{i + 1}")
    wandb.log({"effects": wandb.Image(plt)})
    plt.close("2005/01 effects")


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
