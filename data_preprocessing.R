library(tidyverse)
library(data.table)
library(raster)
library(maptools)
library(proj4)
library(rgdal)
# library(sp)


# Hy-ADS
# load("./data/hyads_unwgted_2005_nobl.RData")
# write_csv(MAP12.2005, "./model_dev_data/hyads_2015_12.csv")
# rm(list=ls())

hyads = read_csv("./data/hyads_2015_12.csv")


# -- Plant data

min_lon = -80.55
min_lat = 39.65
max_lon = -75.25
max_lat = 42.05

target_year = 2015
target_month = 12
n_lags = 6
n_obs = 12

dat = fread("./data/AMPD_Unit_with_Sulfur_Content_and_Regulations_with_Facility_Attributes.csv")
dat = dat[ ,-1]  # drop first column which contains only row number

dat[ ,Fac.ID := dat$Facility.ID..ORISPL.]
dat[ ,uID := paste(dat$Fac.ID, dat$Unit.ID, sep = "_")]
dat[, year_month := paste(Year, Month, sep="_")]
setkeyv(dat, c("uID", "Year", "Month"))
setorderv(dat, c("uID", "Year", "Month"))
dim(dat)  # there are some duplicates
dat <- unique(dat)  # remove duplicates
dim(dat)

df_out = dat %>% 
  as_tibble  %>% 
  rename(fid = Fac.ID,
         so2_tons = SO2..tons.,
         lat = Facility.Latitude.x,
         lon = Facility.Longitude.x,
         month = Month,
         year = Year,
         fuel_type = Fuel.Type..Primary..x,
         state = State.x) %>%
  filter(fuel_type == "Coal") %>% 
  dplyr::select(fid, month, year, so2_tons, lat, lon) %>% 
  na.omit() %>% 
  group_by(fid, month, year) %>% 
  summarise_all(mean)
write_csv(df_out, "./data/so2_data.csv")
write_csv(df_out, "./model_dev_data/so2_data.csv")

power_plant_info = df_out %>% 
  ungroup() %>% 
  dplyr::select(fid, lat, lon) %>% 
  distinct(fid, .keep_all = TRUE)
write_csv(power_plant_info, "./model_dev_data/power_plant_info.csv")

# --- select hy-ads subset for these coal plants
units_hyads = names(hyads)[-1]
fid_hyads = units_hyads %>% 
  stringr::str_split("-") %>% 
  map_chr(~.x[1])
coal_plants_fids = unique(df_out$fid)
# idxs from hyads to keep
col_idxs = fid_hyads %in% coal_plants_fids
hyads_data = hyads[ , c(FALSE, col_idxs)] %>% 
  t() %>% 
  as.data.frame()
names(hyads_data) = as.character(hyads$ZIP)
hyads_data$fid = fid_hyads[col_idxs]
hyads_data = hyads_data %>% 
  group_by(fid) %>% 
  summarise_all(mean) 
write_csv(hyads_data, "./data/hyads_coal.csv")
  
hyads_agg_ = hyads[ , c(TRUE, col_idxs)] %>% 
  as.data.frame() %>% 
  mutate(zipsub = stringr::str_sub(ZIP, end=3)) %>% 
  dplyr::select(-ZIP) %>% 
  group_by(zipsub) %>% 
  summarise_all(mean) 
hyads_agg = hyads_agg_ %>%
  dplyr::select(-zipsub) %>% 
  t() %>% 
names(hyads_agg) = hyads_agg_$zipsub

hyads_aggregated
  

# --- Pollution grid data

fname = "GWRwSPEC_PM25_NA_201512_201512-RH35-NoNegs.asc"
dname = "./data/GWRwSPEC_PM25_NA_201512_201512-RH35-NoNegs.asc/"
grids = raster(paste0(dname, fname))
values(grids)[values(grids) < 0] = NA
sp_data = rasterToPoints(grids) %>% 
  as_data_frame %>% 
  mutate(year = 2015) %>% 
  mutate(month = 12) %>% 
  rename(pm25 = starts_with("GWR"),
         lon = x,
         lat = y)
write_csv(sp_data, "data/grid_pm25.csv")

sp_data_sub = sp_data %>% 
  dplyr::filter(min_lon <= lon,
                lon <= max_lon,
                min_lat <= lat,
                lat <= max_lat)
write_csv(sp_data_sub, "./model_dev_data/grid_pm25_subset.csv")
