# Install required R packages for goalie xG pipeline
pkgs <- c("tidyverse", "httr2", "jsonlite", "tidymodels", "xgboost", "arrow", "duckdb", "DBI", "cli")
install.packages(pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)])
