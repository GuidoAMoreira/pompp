#' @importFrom parallel detectCores
#' @export
fit_pompp <- function(beta, delta, lambda, bPriorPar, dPriorPar, lPriorPar,
                      covariatesMatrix, area, marksMuPriorPar, marksNuggetPriorPar,
                      marksShapePriorPar, observedCovariatesMatrix, observedMarks,
                      observedPositions, intensityColumns, observabilityColumns,
                      xIntensityColumns, xObservabilityColumns, neighborhoodSize,
                      longitudeColumns, latitudeColumns, burnin, thin, iter) {
  cppPOMPP(
    beta, delta, lambda, "", "", "", bPriorPar, dPriorPar, lPriorPar$a,
    lPriorPar$b, "", covariatesMatrix, area, "", marksMuPriorPar$mean,
    marksMuPriorPar$variance, marksNuggetPriorPar$a, marksNuggetPriorPar$b,
    marksShapePriorPar$a, marksShapePriorPar$b, observedCovariatesMatrix,
    observedMarks, observedPositions, intensityColumns, observabilityColumns,
    xIntensityColumns, xObservabilityColumns, neighborhoodSize, longitudeColumns, latitudeColumns,
    burnin, thin, iter, parallel::detectCores(), TRUE
  )
}

