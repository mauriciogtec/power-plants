#%%
from typing import Tuple
import numpy as np
from tqdm import tqdm
import pickle as pkl
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


min_lon = -80.55
min_lat = 39.65
max_lon = -75.25
max_lat = 42.05

# %%

years = range(2000, 2016)
months = range(1, 13)

dname = "./data/PM25/PM25"
root = "V4NA03_PM25_NA"

# first 6 columns in raster files
ncols = 9300
nrows = 4550
XLLCENTER = -137.995
YLLCENTER = 22.505
CELLSIZE = 0.01
NODATA_value = -999.9

# deternmine boolean mask of grid values to hold
xcoords = XLLCENTER + CELLSIZE * np.arange(ncols)
ycoords = YLLCENTER + CELLSIZE * np.arange(nrows)

xcoords_in = (ycoords > min_lat) & (ycoords < max_lat)
ycoords_in = (xcoords > min_lon) & (xcoords < max_lon)

# %%

# data = {}
# for y in years:
#     for m in months:
#         ym = f"{y}{m:02d}"
#         bname = f"{root}_{ym}_{ym}-RH35-NoNegs.asc"
#         fname = f"{dname}/{bname}/{bname}"
#         print(f"Reading {fname}...")
#         grid = np.loadtxt(fname, skiprows=6)
#         data[ym] = grid[xcoords_in, :][:, ycoords_in]
# with open("data/grid_phil.pkl", "wb") as io:
#     pkl.dump(data, io)

# %%
with open("data/grid_phil.pkl", "rb") as io:
    data = pkl.load(io)
ym2index = {
    ym: i for i, ym in enumerate(sorted(data.keys()))
    if int(ym[:4]) <= 2014
}

# %%
t = len(ym2index)
na_val = -999.9
r, c = list(data.values())[0].shape
y = np.zeros((r, c, t), dtype=np.float32)
miss = np.zeros((r, c, t), dtype=np.bool)
for ym, val in data.items():
    if ym not in ym2index:
        continue
    ti = ym2index[ym]
    miss[:, :, ti] = (val == na_val)
    val[val == na_val] = 0.0
    y[:, :, ti] = val

# %%  Part 2
pp = pd.read_csv("model_dev_data/so2_data.csv")
pp['ym'] = pp.year.astype(str) + pp.month.apply(lambda x: f"{x:02d}")
fid2index = {fid: i for i, fid in enumerate(sorted(set(pp.fid)))}
n = len(fid2index)
X = np.zeros((n, t), dtype=np.float32)
skipped_dates = set()
for _, row in pp.iterrows():
    i = fid2index[row['fid']]
    if row['ym'] not in ym2index:
        skipped_dates.add(row['ym'])
        continue
    ti = ym2index[row['ym']]
    X[i, ti] = row['so2_tons']
print("skipped dates:", skipped_dates)

frac_obs = (X > 0.0).sum(1) / t
frac_obs_per_date = (X > 0.0).sum(0) / n
enough = frac_obs >= (thresh := .99)
fid2index_new = {k: v for k, v in fid2index.items() if enough[v]}
print(f"There are {sum(enough)} power plants with at least {int(100*thresh)}% data")

#%%
locs = pp[["fid", "lon", "lat"]].drop_duplicates()
inrows = [i for i in range(locs.shape[0]) if locs.fid.iloc[i] in fid2index_new]
locs = locs.iloc[inrows]
fid = locs.fid.values
locs = locs[["lon", "lat"]].values


# %% use missing data imputation
X1 = np.log(X[enough, :])
if thresh < 1.0:
    imp = IterativeImputer(max_iter=10, random_state=0)
    IterativeImputer(random_state=0)
    X1 = np.log(X[enough, :])
    X1[np.isinf(X1)] = np.nan
    imp.fit(X1)
    X1 = imp.transform(X1)
X = np.exp(X1)

# %%
savedict = dict(
    X=X,
    y=y,
    ym=sorted(ym2index.keys()),
    locs=locs,
    xcoords=xcoords,
    ycoords=ycoords,
)
np.savez("model_dev_data/phil.npz", **savedict)

# %%
