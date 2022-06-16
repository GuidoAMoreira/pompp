library(pompp)
set.seed(123)
setwd("/home/anthorg/Documents/Work/pkg/pompp")

#### Read data ####
simulated <- readRDS("tests/data/simulatedData.rds")

phi <- 0.3
sigma2 <- 2

time <- Sys.time()
runMCMC <- fit_pompp(
  rep(0, 4), rep(0, 4), 10,
  list(mean = rep(0, 4), covariance = 100 * diag(4)),
  list(mean = rep(0, 4), covariance = 100 * diag(4)),
  list(a = 0.001, b = 0.001),
  cbind(simulated$Xint, simulated$Xobs, simulated$grid),
  1, 0, 1, list(mean = 0, variance = 100),
  list(a = 0.001, b = 0.001),
  cbind(simulated$Z_X, simulated$W_X),
  simulated$observedMarks, simulated$X,
  1:3, 4:5, 1:3, 4:5,
  sqrt(2), sigma2, phi,
  20, 6, 7,
  50000, 10, 100000, 6
)
print(Sys.time() - time)
# sink("output.txt")
# runMCMC <- fit_pompp(
#   rep(0, 4), rep(0, 4), 10,
#   list(mean = rep(0, 4), covariance = 10 * diag(4)),
#   list(mean = rep(0, 4), covariance = 10 * diag(4)),
#   list(a = 0.001, b = 0.001),
#   cbind(simulated$Xint, simulated$Xobs, simulated$grid),
#   1, 0, 1, list(mean = 0, variance = 100),
#   list(a = 0.001, b = 0.001),
#   cbind(simulated$Z_X, simulated$W_X),
#   simulated$observedMarks, simulated$X,
#   1:3, 4:5, 1:3, 4:5,
#   sqrt(2), sigma2, phi,
#   5, 6, 7,
#   0, 1, 5, 1
# )
# sink()

saveRDS(runMCMC, "tests/data/mcmcResult.rds")
# saveRDS(runMCMC, "~/Dropbox/Work/PosPreferential/pompp/tests/mcmcResult.rds")
