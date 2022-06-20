library(geoR)
set.seed(123)
setwd("/home/anthorg/Documents/Work/pkg/pompp")

#### True values ####
source("tests/trueValues.R")
nb <- length(beta)
nd <- length(delta)

#### Actual simulation ####
totalPoints <- rpois(1, lambdaStar)
totalPositions <- cbind(runif(totalPoints), runif(totalPoints))

#### Intensity ####
Xint <- cbind(grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
    cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data,
grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
             cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data,
grf(1, grid = rbind(totalPositions, squareGrid), cov.model = "exponential",
             cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data)

unifs1 <- runif(totalPoints)
intensitySelection <- which(log(unifs1) - log1p(-unifs1) <=
  beta[1] + Xint[1:totalPoints, ] %*% beta[2:nb])
intensitySelected <- length(intensitySelection)

occurrencesPositions <- totalPositions[intensitySelection, ]
occurrencesXint <- Xint[intensitySelection, ]
Xobs <- cbind(grf(1, grid = rbind(occurrencesPositions, squareGrid), cov.model = "exponential",
                  cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data,
              grf(1, grid = rbind(occurrencesPositions, squareGrid), cov.model = "exponential",
                  cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data)
S <- grf(1, grid = occurrencesPositions, cov.model = "exponential",
         cov.pars = c(sigma2, phi), kappa = 0, method = "cholesky")$data

unifs2 <- runif(intensitySelected)
observabilitySelection <- which(log(unifs2) - log1p(-unifs2) <=
  delta[1] + Xobs[1:intensitySelected, ] %*% delta[2:nd] + gamma * S)
observabilitySelected <- length(observabilitySelection)

marks <- rgamma(intensitySelected, shape, shape / exp(mu + S + rnorm(intensitySelected, 0, sqrt(nugget))))

observedPositions <- occurrencesPositions[observabilitySelection, ]
observedXint <- occurrencesXint[observabilitySelection, ]
observedXobs <- Xobs[observabilitySelection, ]
observedMarks <- marks[observabilitySelection]

saveRDS(
  list(Xint = Xint[-(1:totalPoints), ], Xobs = Xobs[-(1:intensitySelected),],
       grid = squareGrid, X = observedPositions,
       Z_X = observedXint, W_X = observedXobs,
       observedMarks = observedMarks,
       Xp = occurrencesPositions[-observabilitySelection, ],
       Z_Xp = occurrencesXint[-observabilitySelection, ],
       W_Xp = Xobs[-observabilitySelection, ],
       unobservedMarks = marks[-observabilitySelection]),
  "tests/data/simulatedData.rds"
)
