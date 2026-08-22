#!/usr/bin/env Rscript
# Bootstrap installer for bLSmodelR and required deps.
# Run once per R environment: Rscript install_bls.R
repos <- getOption("repos")
if (is.null(repos) || repos["CRAN"] == "@CRAN@") {
  repos <- c(CRAN = "https://cloud.r-project.org")
  options(repos = repos)
}

needed <- c("jsonlite", "data.table", "sp", "Rcpp", "RcppArmadillo")
missing <- needed[!needed %in% rownames(installed.packages())]
if (length(missing)) install.packages(missing)

if (!"bLSmodelR" %in% rownames(installed.packages())) {
  if (!"remotes" %in% rownames(installed.packages())) install.packages("remotes")
  remotes::install_github("ChHaeni/bLSmodelR")
}

suppressMessages(library(bLSmodelR))
cat(sprintf("bLSmodelR %s installed OK\n", as.character(packageVersion("bLSmodelR"))))
