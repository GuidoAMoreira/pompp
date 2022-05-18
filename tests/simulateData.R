library(geoR)
set.seed(123)

#### True values ####
source("tests/trueValues.R")

#### Actual simulation ####
totalPoints <- rpois(1, lambdaStar)
totalPositions <- c(runif(totalPoints), runif(totalPoints))

#### Intensity ####
Xint1 <- grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
    cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")
Xint2 <- grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
             cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")
Xint3 <- grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
             cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")
