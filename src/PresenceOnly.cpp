#include <RcppEigen.h>
#include "include/PresenceOnly.hpp"
#include "include/BackgroundVariables.hpp"
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

double PresenceOnly::updateLambdaStar() {
  double a = aL + x.rows() + xprime.rows() + u.rows(),
    b = bL + area;
  lambdaStar = R::rgamma(a, 1 / b);

  return - lambdaStar * b + (a - 1) * log(lambdaStar);
}

double PresenceOnly::sampleProcesses() {
  // Determining number of points in X' and U
  /*
   * Technically, the correct code would sample from a truncated Poisson.
   * This approximation should only be problematic if the data comes from a
   * homogeneous process, which would defeat the purpose of the analysis anyway
   * so there is no loss.
   */
  bool stillSampling;

  long totalPoints = R::rpois(lambdaStar * area);
  if (totalPoints <= x.rows()) {
    xprime.resize(0, 2);
    u.resize(0, 2);
    xxprimeIntensity.resize(x.rows(), beta->getSize() - 1);
    xprimeObservability.resize(0, delta->getSize() - 1);
    uIntensity.resize(0, beta->getSize() - 1);
    bkg->startGPs();
    return 0.;
  }

  // Sampling from X' and U
  totalPoints -= x.rows(); // Number of points associated to them
  double p, q, uniform;
  long accXp = 0, accU = 0, currentAttempt;
  Eigen::VectorXd candidate;
  Eigen::MatrixXd storingCoords(totalPoints, 2); // Put X' on top and U on bottom
  bkg->startGPs();
  for (int i = 0; i < totalPoints; i++) {
    currentAttempt = 0;
    stillSampling = true;
    while (stillSampling && ++currentAttempt < MAX_ATTEMPTS_GP) {
      R_CheckUserInterrupt();
      candidate = bkg->getRandomPoint();
      uniform = log(R::runif(0, 1));
      q = beta->link(bkg->getVarVec(candidate, INTENSITY_VARIABLES));
      if (uniform > q) { // Assign to U
        storingCoords.row(totalPoints - ++accU) = candidate.transpose();
        bkg->acceptNewPoint(INTENSITY_VARIABLES);
        stillSampling = false;
      } else {
        p = delta->link(bkg->getVarVec(candidate, OBSERVABILITY_VARIABLES));
        if (uniform > p + q) { // Assign to X'
          storingCoords.row(accXp++) = candidate.transpose();
          bkg->acceptNewPoint(OBSERVABILITY_VARIABLES);
          stillSampling = false;
        } // Else discard candidate and try again.
      }
    }
    if (currentAttempt == MAX_ATTEMPTS_GP) Rf_warning("GP sampling reached max attempts.");
  }
  bkg->setGPinStone();


  xprime.resize(accXp, 2);
  u.resize(accU, 2);

  xxprimeIntensity.resize(x.rows() + accXp, beta->getSize() - 1);
  xprimeObservability.resize(accXp, delta->getSize() - 1);

  if (accXp) {
    xxprimeIntensity.bottomRows(accXp) =
      bkg->getVarMat(storingCoords.topRows(accXp),INTENSITY_VARIABLES);
    xprimeObservability = bkg->getVarMat(storingCoords.topRows(accXp), OBSERVABILITY_VARIABLES);
  }

  uIntensity.resize(accU, beta->getSize() - 1);
  if (accU)
    uIntensity = bkg->getVarMat(storingCoords.bottomRows(accU), INTENSITY_VARIABLES);

  return - lgamma(accXp + 1) - lgamma(accU + 1);
}

inline double PresenceOnly::applyTransitionKernel() {
  double out, privateOut1, privateOut2;
  out = sampleProcesses() + updateLambdaStar();
#ifdef _OPENMP
#pragma omp parallel
#endif
{
#ifdef _OPENMP
#pragma omp sections nowait
#pragma omp section
#endif
  privateOut1 = beta->sample(xxprimeIntensity, uIntensity);
#ifdef _OPENMP
#pragma omp section
#endif
  privateOut2 = delta->sample(xObservability, xprimeObservability);
}
  return out + privateOut1 + privateOut2;
}
