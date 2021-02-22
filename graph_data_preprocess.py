# %%
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import pickle as pkl


# %% read spatial graph
avs = pd.read_csv("./model_dev_data/zip_averages.csv", dtype=dict(lab=str))
adj = pd.read_csv(
    "./model_dev_data/zip_adjacency.csv",
    dtype=dict(src_lab=str, tgt_lab=str)
)
controls = pd.read_csv(
    "./model_dev_data/covars_averages.csv", dtype=dict(lab=str)
)
labs0 = avs.lab


# %% read hyads to filter zip codes not in hyads
m = 12
f = f"model_dev_data/hyads/2015/{m:02d}.csv"
hyads = pd.read_csv(f, dtype=dict(ZIP=str))


# %%
hasdata = (
    avs.values[:, 1:].min(-1) > 0
    & avs.lab.isin(controls.lab).values
    & avs.lab.isin(hyads.ZIP).values
)
avs = avs.iloc[hasdata, :]
controls = controls.iloc[controls.lab.isin(avs.lab).values, :]


# %%
adj = adj[
    adj.src_lab.isin(avs.lab)
    & adj.tgt_lab.isin(avs.lab)
]


# %%
g = nx.Graph()
g.add_edges_from(zip(adj.src_lab, adj.tgt_lab))

# %%
N = g.number_of_nodes()
E = g.number_of_edges()

print(f"Graph with {N} nodes and {E} edges")

# %%
cc1 = sorted(nx.connected_components(g), key=len)[-1]
g1 = nx.subgraph(g, cc1)

# %%
N = g1.number_of_nodes()
E = g1.number_of_edges()

print(f"Graph with {N} nodes and {E} edges")

# let's take only the largex conntected component
# %%

nodemap = {x: i for i, x in enumerate(g1.nodes())}
nodes = list(nodemap.keys())

adj_idx = adj[adj.src_lab.isin(nodes) & adj.tgt_lab.isin(nodes)]

# %%

avs_idx = avs.set_index("lab").loc[nodes]

from_date = "200001"
to_date = "201412"
nt = 180
Y = np.zeros((N, nt))

for t in range(nt):
    year = 2000 + t // 12
    month = t % 12 + 1
    ym = f"{year}{month:02d}"
    var = f"so4_{ym}_{ym}"
    Y[:, t] = avs_idx[var]


# %%
controls_idx = controls.set_index("lab").loc[nodes]
vnames = ["temp", "apcp", "rhum", "vwnd"]
ncovars = len(vnames)
C = np.zeros((N, ncovars, nt))
for t in range(nt):
    year = 2000 + t // 12
    month = t % 12 + 1
    ym = f"{year}{month:02d}"
    for j, v in enumerate(vnames):
        var = f"so4_{ym}_{v}"
        C[:, j, t] = controls_idx[var]


# %%

pp = pd.read_csv("model_dev_data/so2_data.csv")
pp['ym'] = pp.year.astype(str) + pp.month.apply(lambda x: f"{x:02d}")

pp_idx = pp.set_index(["ym", "fid"])["so2_tons"]
pp_idx

locs = pp[["fid", "lon", "lat"]].drop_duplicates()


# %%
fids = np.array(sorted(set(pp.fid)))
P = len(fids)


# %%
X = np.zeros((P, nt))

for t in range(nt):
    year = 2000 + t // 12
    month = t % 12 + 1
    ym = f"{year}{month:02d}"
    tmp = pp_idx[ym]
    for j, f in enumerate(fids):
        if f in tmp.index:
            X[j, t] = tmp[f]


# %%
missing_per_fid = (X == 0).sum(1)
missing_per_year = (X == 0).sum(0)

# %%

sns.lineplot(
    x=[date(2000 + t // 12, t % 12 + 1, 1) for t in range(180)],
    y=missing_per_year / len(fids)
)
plt.title("missing fraction")

# %%
sns.distplot(missing_per_fid, kde=False)
ff = np.mean(missing_per_fid == 0)
plt.title(f"Fraction of fids with full data {100 * ff:.2f}%")

# %%
fulldata = missing_per_fid == 0
fulldata_fids = fids[fulldata]
Xfull = X[fulldata, :]
locsfull = locs.iloc[fulldata, :]


# %%  coords and dists
zip_coords = pd.read_csv("model_dev_data/zip_coords.csv", dtype=dict(ZIP=str))
zip_coords = zip_coords.set_index("ZIP").loc[nodes]


# %%
zip_to_pp_dists = np.loadtxt(
    "model_dev_data/zip_to_pp_dists.csv",
    delimiter=","
)
zip_to_pp_dists_in = zip_to_pp_dists[:, fulldata]
nodesin = np.array([lab in nodemap for lab in labs0])
zip_to_pp_dists_in = zip_to_pp_dists_in[nodesin]

# %%

traindata = dict(
    Y=Y.transpose(),
    X=Xfull.transpose(),
    C=C.transpose(2, 0, 1),
    fid=fulldata_fids,
    locs=locsfull,
    min_year=2000,
    min_month=1,
    adj=adj_idx,
    nodes=nodes,
    zip_coords=zip_coords,
    zip_to_pp_dists=zip_to_pp_dists_in
)


# %%
with open("model_dev_data/graph_training_data.pkl", "wb") as io:
    pkl.dump(traindata, io, protocol=pkl.HIGHEST_PROTOCOL)
# %%
