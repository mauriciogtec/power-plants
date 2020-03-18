library(tidyverse)
library(data.table)
library(raster)
library(maptools)
library(proj4)
library(data.table)
library(zipcode)
library(rgdal)
library(sp)

min_lon = -80.55
min_lat = 39.65
max_lon = -75.25
max_lat = 42.05
yr = 2015

# -- Plant data

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
  as_data_frame %>% 
  rename(id = uID,
         so2_tons = SO2..tons.,
         lat = Facility.Latitude.x,
         lon = Facility.Longitude.x,
         month = Month,
         year = Year,
         fuel_type = Fuel.Type..Primary..x,
         state = State.x) %>%
  dplyr::select(id, month, year, so2_tons,
                lat, lon, state, fuel_type) %>% 
  na.omit()
write_csv(df_out, "./data/so2_data.csv")

df_out_sub = df_out %>% 
  dplyr::filter(min_lon <= lon,
                lon <= max_lon,
                min_lat <= lat,
                lat <= max_lat) %>% 
  dplyr::filter(yr - 2 <= year,
                year <= yr)
write_csv(df_out_sub, "./model_dev_data/so2_data_subset.csv")


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
