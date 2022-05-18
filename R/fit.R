#' @importFrom parallel detectCores
#' @export
fit_pompp <- function(beta, delta, lambda, bPriorPar, dPriorPar, lPriorPar,
                      covariatesMatrix, area, mu, nugget, shape,
                      marksMuPriorPar, marksNuggetPriorPar, marksShapePriorPar,
                      observedCovariatesMatrix, observedMarks, observedPositions,
                      intensityColumns, observabilityColumns,
                      xIntensityColumns, xObservabilityColumns,
                      maxDist, sigma2, phi, neighborhoodSize,
                      longitudeColumns, latitudeColumns, burnin, thin, iter) {
  cppPOMPP(
    beta, delta, lambda, "", "", "", bPriorPar, dPriorPar, lPriorPar$a,
    lPriorPar$b, "", covariatesMatrix, area, "", mu, nugget, shape, marksMuPriorPar$mean,
    marksMuPriorPar$variance, marksNuggetPriorPar$a, marksNuggetPriorPar$b,
    marksShapePriorPar$a, marksShapePriorPar$b, observedCovariatesMatrix,
    observedMarks, observedPositions, intensityColumns - 1, observabilityColumns - 1,
    xIntensityColumns - 1, xObservabilityColumns - 1, maxDist, sigma2, phi,
    as.integer(neighborhoodSize), as.integer(longitudeColumns - 1), as.integer(latitudeColumns - 1),
    as.integer(burnin), as.integer(thin), as.integer(iter), parallel::detectCores(), TRUE
  )
}

