library(geoR)

#### True values ####
beta <- c(1, -1, 2, -1.5)
delta <- c(-1, -1, -2)
lambdaStar <- 1000
gamma <- 2
shape <- 1.5
nugget <- 0.5
mu <- 5

#### Meta parameters ####
gridSize <- 100
squareGrid <- expand.grid(seq(0, 1, len = gridSize), seq(0, 1, len = gridSize))

#### Actual simulation ####
totalPoints <- rpois(1, lambdaStar)
totalPositions <- c(runif(totalPoints), runif(totalPoints))

#### Intensity ####


grf(1, grid = pts, cov.model = "exponential",
    cov.pars = c(sigma2, covpar[1]), kappa = 0, method = "cholesky")
