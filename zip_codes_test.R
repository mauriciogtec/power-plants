library(parallel)
library(tidyverse)
# library(rgeos)
# library(maptools)
# library(geosphere)
# library(progress)
library(raster)
library(rgdal)
# library(viridis)
# library(pbapply)
library(stringr)

# Cluster for parallel computing
num_processes = 8

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
# pblapply
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
  }#,
  # cl=cl
)
stopCluster(cl)


# Read .tif files
datadir = "./data/SO4/"
grids = list.files(
  datadir,
  pattern = "tif$",
  recursive = TRUE,
  include.dirs = FALSE
)
paths_tif = paste0(datadir, grids)


# Find the raster mean per polygon

print("Computing stats...")
zipshp = "data/tl_2016_us_zcta510/tl_2016_us_zcta510.shp"
zips = readOGR(zipshp)
projection(zips) = crswgs84

# zips = zips[1:1000, ]/

cl = makeCluster(num_processes)
overwrite = FALSE
clusterExport(cl, c("zips", "overwrite", "crswgs84", "datadir"))

# pblapply
parLapply(cl,
  paths_tif,
  function(p) {
    tgt_dir = "model_dev_data/so4/"
    basefile = stringr::str_split(p, "/")[[1]]
    basefile = basefile[length(basefile)]
    basefile = stringr::str_sub(basefile, end=-5)
    path_rds = paste0(tgt_dir, basefile, "_zipcode_mean.rds")
    if (overwrite || !file.exists(path_rds)) {
      rast = raster::raster(p)
      raster::projection(rast) = crswgs84
      # the resolution is too high to find each point in
      # polygon, so first average uniformly by a factor of 4
      rast = raster::aggregate(rast, method="", fact=4, na.rm=TRUE)
      # average by polygons
      results = raster::extract(rast, zips, fun=mean, na.rm=TRUE)
      print(paste("Finished", path_rds))
      saveRDS(results, path_rds) 
      rm(results)
    }
  }#,
  # cl=cl
)
