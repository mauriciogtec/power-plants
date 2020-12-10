#%%
import numpy as np
import torch
import pandas as pd

# %%
y = pd.read_csv("model_dev_data/grid_pm25_subset.csv")
X = pd.read_csv("model_dev_data/so2_data_subset.csv")

# %%

# 1. Convert to grid
lon_min, lon_max = y.lon.min(), y.lon.max()
lat_min, lat_max = y.lat.min(), y.lat.max()

delta = 0.01
eps = 1e-6

nrow = int((lat_max - lat_min) / delta)
ncol = int((lon_max - lon_min) / delta)
plants_ids = X.id.unique()
nplants = len(plants_ids)

y["t"] = y.year.astype(str) + "_" + y.month.astype(str)
t_ids = {x: i for x, i in enumerate(y.t.unique())

# %%
len(X.id.unique())



# %%
