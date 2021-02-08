library(disperseR)


# why would the default be the working directory?
disperseR::create_dirs(getwd())

# function is not documented
# need to build the meteo_dir files
# why metfiles?

disperseR::get_data(
  data="metfiles",
  start.year = "2000",
  start.month = "01",
  end.year = "2015",
  end.month = "12"
)
