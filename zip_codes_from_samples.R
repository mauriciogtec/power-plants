library(tidyverse)
library(raster)
library(rgdal)
library(stringr)

# Define the projection to be used
crswgs84=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

# 1 convert asc writer to .tif
# this should be done in the download data module
# Find asc rasters  

datadir = "./data/SO4/"
grids = list.files(
  datadir,
  pattern = "asc$",
  recursive = TRUE,
  include.dirs = FALSE
)

if (length(grids) > 0) {
  paths = paste0(datadir, grids)
} else {
  paths = character(0)
}


# Convert all rasters to .tiff if necessary
print("Converting .asc to .tif...")
cl = makeCluster(num_processes)
# pblapply(
parLapply(cl,
  paths,
  function(p) {
    path_tif = paste0(stringr::str_sub(p, end=-5), ".tif")
    crswgs84=sp::CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    if (!file.exists(path_tif)) {
      rast = raster::raster(p)
      raster::projection(rast) = crswgs84
      raster::writeRaster(rast, path_tif, overwrite=TRUE)
      rm(rast)
    }
    file.remove(p)
  # }, cl=cl
  }
)
stopCluster(cl)


# Read .tif files
datadir = "./data/SO4/"
grids = list.files(
  datadir,
  pattern = "NoNegs\\.tif$",
  recursive = TRUE,
  include.dirs = FALSE
)
paths_tif = paste0(datadir, grids)
rast = raster::stack(paths_tif)


# read sample points
pts = read_csv("data/zip_samples.csv")


# extract
matches = raster::extract(rast, select(pts, lon, lat))
