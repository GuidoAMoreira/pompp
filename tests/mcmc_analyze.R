library(pompp)
library(dplyr)
library(ggplot2); theme_set(theme_bw())
library(bayesplot)
library(GGally)
library(gridExtra)
library(glue)

source("tests/trueValues.R")
runMCMC <- readRDS("tests/data/mcmcResult.rds"); mcmc_mat <- do.call(cbind, runMCMC)
colnames(mcmc_mat)[1:8] <- c(paste0("beta", 0:3),
                             paste0("delta", 0:2), "gamma")

length(runMCMC)
summary(mcmc_mat)
summary(mcmc_mat)
as.data.frame(mcmc_mat) %>% mcmc_trace()
as.data.frame(mcmc_mat) %>% mcmc_dens()
as.data.frame(mcmc_mat) %>% ggpairs()

# Comparing with true values
simulated <- readRDS("tests/data/simulatedData.rds") # For marks true values
allLabels <- list(
  expression(beta[0]), expression(beta[1]), expression(beta[2]), expression(beta[3]),
  expression(delta[0]), expression(delta[1]), expression(delta[2]), expression(gamma),
  expression(lambda^"*"), expression(mu), expression(tau^2), "nU", "nXp",
  "Sum of Z'", "Variance of Z'", "Sum of Z", "Variance of Z", "Log-posterior"
)
allTrues <- c(
  beta, delta, gamma, lambdaStar, mu, nugget, NA, nrow(simulated$Xp),
  sum(simulated$unobservedMarks),
  mean(simulated$unobservedMarks ^ 2) - mean(simulated$unobservedMarks) ^ 2,
  sum(simulated$unobservedMarks) + sum(simulated$observedMarks),
  sum(simulated$unobservedMarks ^ 2) + sum(simulated$observedMarks ^ 2) /
    (nrow(simulated$Xp) + nrow(simulated$X)) -
    ((sum(simulated$unobservedMarks) + sum(simulated$observedMarks)) /
    (nrow(simulated$Xp) + nrow(simulated$X))) ^ 2,
  NA
)
graphs <- list()
for (i in 1:ncol(mcmc_mat)) {
  graphs[[i]] <-
    ggplot(data.frame(var = mcmc_mat[, i]), aes(var)) +
      geom_density() + xlab(allLabels[[i]]) +
    geom_vline(xintercept = allTrues[i], col = "red", size = 1.2)
}
grid.arrange(grobs = graphs, ncol = 4)

ggcorr(as.data.frame(mcmc_mat))
ggpairs(as.data.frame(mcmc_mat[, c(2, 3, 9, 12, 13)]))
