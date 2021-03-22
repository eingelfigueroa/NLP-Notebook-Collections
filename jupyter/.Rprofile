# Warning: .Rprofile  is no longer automatically updated
#  by DSWB for existing users as of 2016-Sept-22
.First <- function() {
  if (Sys.getenv("R_CMD") == "") {
    options(repos = structure(c(CRAN = "https://cran.r-project.org/")))
    Sys.setenv('SPARKR_SUBMIT_ARGS'='"--packages" "com.databricks:spark-csv_2.10:1.5.0" "sparkr-shell"')
    .libPaths(c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib"), .libPaths()))
    Sys.setenv("R_LIBS_USER" = "/resources/common/R/Library")
    .libPaths(c(file.path(Sys.getenv("R_LIBS_USER")), .libPaths()))
    library(SparkR)
    sc <<- sparkR.init()
  }
}
