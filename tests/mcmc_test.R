library(pompp)
set.seed(123)
setwd("/home/anthorg/Documents/Work/pkg/pompp")

#### Read data ####
simulated <- readRDS("tests/data/simulatedData.rds")

runMCMC <- fit_pompp(
  rep(0, 4), rep(0, 4), 10,
  list(mean = rep(0, 4), Sigma = 10 * diag(4)),
  list(mean = rep(0, 4), Sigma = 10 * diag(4)),
  list(a = 0.001, b = 0.001),
  cbind(simulated$Xint, simulated$Xobs, grid),
  1, list(mean = 0, variance = 100),
  list(a = 0.001, b = 0.001),
  list(a = 0.001, b = 0.001),
  cbind(simulated$Z_X, simulated$W_X),
  simulated$observedMarks, simulated$X,
  1:3, 4:5, 1:3, 4:5,
  20, 7, 8,
  10, 1, 100
)
