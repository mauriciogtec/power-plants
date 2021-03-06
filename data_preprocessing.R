library(tidyverse)
library(data.table)
library(raster)
library(maptools)
library(proj4)
library(rgdal)
# library(sp)


# Hy-ADS
# load("./data/hyads_unwgted_2005_nobl.RData")
# write_csv(MAP12.2005, "./data/hyads_2015_12.csv")
# rm(list=ls())

# hyads = read_csv("./data/hyads_2015_12.csv")


# -- Plant data

min_lon = -80.55
min_lat = 39.65
max_lon = -75.25
max_lat = 42.05

target_years = 2015
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
dat = unique(dat)  # remove duplicates
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
write_csv(df_out, "./model_dev_data/so2_data.csv")

# df_out = read_csv("./model_dev_data/so2_data.csv")
# 
# power_plant_info = df_out %>% 
#   ungroup() %>% 
#   dplyr::select(fid, lat, lon) %>% 
#   distinct(fid, .keep_all = TRUE)
# 
# write_csv(power_plant_info, "./model_dev_data/power_plant_info.csv")
# 
# # --- select hy-ads subset for these coal plants
# units_hyads = names(hyads)[-1]
# units_hyads = units_hyads %>% 
#   stringr::str_split("-") %>% 
#   map_chr(~.x[1])
# 
# coal_plants_s = unique(df_out$fid)
# # idxs from hyads to keep
# col_idxs = _hyads %in% coal_plants_s
# hyads_data = hyads[ , c(FALSE, col_idxs)] %>% 
#   t() %>% 
#   as.data.frame()
# names(hyads_data) = as.character(hyads$ZIP)
# hyads_data$ = _hyads[col_idxs]
# hyads_data = hyads_data %>% 
#   group_by() %>% 
#   summarise_all(sum)  # adding instead of mean because particles are spread accross units 
# write_csv(hyads_data, "./data/hyads_coal.csv")
#   
# hyads_agg_ = hyads[ , c(TRUE, col_idxs)] %>% 
#   as.data.frame() %>% 
#   mutate(zipsub = stringr::str_sub(ZIP, end=3)) %>% 
#   dplyr::select(-ZIP) %>% 
#   group_by(zipsub) %>% 
#   summarise_all(mean) 
# hyads_agg = hyads_agg_ %>%
#   dplyr::select(-zipsub) %>% 
#   t() %>% 
#   as.data.frame() %>% 
#   as_tibble() %>% 
#   add_column( = _hyads[col_idxs], .before=0) %>% 
#   group_by() %>% 
#   summarise_all(mean) 
# names(hyads_agg)[-1] = hyads_agg_$zipsub
# write_csv(hyads_agg, "./model_dev_data/hyads_zipcode_3digits.csv")
#   

# --- Pollution grid data=

# currently I'm only saving a substet of the da

years = 2000:2015
months = 1:12L

sp_data_sub = list()
i = 1
dname = "./data/PM25/PM25"
# root = "GWRwSPEC_PM25_NA"
root = "V4NA03_PM25_NA"
for (yr in years) {
  for (m in months) {
    file = sprintf("%s_%d%02d_%d%02d-RH35-NoNegs.asc", root, yr, m, yr, m)
    fpath = paste(dname, file, file, sep="/")
    grids = raster(fpath)
    # values(grids)[values(grids) < 0] = NA
    sp_data = rasterToPoints(grids) %>% 
      as_tibble %>% 
      mutate(year = yr) %>% 
      mutate(month = m) %>% 
      rename(pm25 = starts_with(root),
             lon = x,
             lat = y) %>% 
      dplyr::filter(min_lon <= lon,
                    lon <= max_lon,
                    min_lat <= lat,
                    lat <= max_lat)
    sp_data_sub[[i]] = sp_data
    i = i + 1
    print(sprintf("Finished %s/%s", year, month))
    ftgt = sprintf("./model_dev_data/grid_subset/%s-%s.csv", year, month)
    write_csv(sp_data_sub, ftgt)
  }
}

# sp_data_sub = bins_rows(sp_data_sub)
