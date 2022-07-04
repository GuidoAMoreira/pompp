beta <- c(-2, -1, 2, -1.5)
delta <- c(-1, -1, -2)
lambdaStar <- 6000
gamma <- 2
shape <- 1.5
nugget <- 0.5
mu <- 5
phi <- 0.3
sigma2 <- 2

#### Meta parameters ####
gridSize <- 100
squareGrid <- as.matrix(expand.grid(seq(0, 1, len = gridSize), seq(0, 1, len = gridSize)))
