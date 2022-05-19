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
  xObservability.col(xObservability.cols() - 1) =
    bkg->getGPfull(OBSERVABILITY_VARIABLES).head(x.rows());

  long totalPoints = R::rpois(lambdaStar * area);
  if (totalPoints <= x.rows()) {
    xprime.resize(0, 2);
    u.resize(0, 2);
    xxprimeIntensity = xIntensity;
    xprimeObservability.resize(0, delta->getSize() - 1);
    uIntensity.resize(0, beta->getSize() - 1);
    bkg->startGPs(0);
    marksPrime = Eigen::VectorXd(0);
    marksExpected.conservativeResize(x.rows());
    return 0.;
  }

  // Sampling from X' and U
  marksExpected.conservativeResize(totalPoints);
  totalPoints -= x.rows(); // Number of points associated to them
  double p, q, uniform;
  long accXp = 0, accU = 0, currentAttempt;
  Eigen::VectorXd candidate;
  Eigen::MatrixXd storingCoords(totalPoints, 3); // Put X' on top and U on bottom
  marksPrime = Eigen::VectorXd(totalPoints);
  bkg->startGPs(totalPoints);
  for (int i = 0; i < totalPoints; i++) {
    currentAttempt = 0;
    stillSampling = true;
    while (stillSampling && ++currentAttempt < MAX_ATTEMPTS_GP) {
      R_CheckUserInterrupt();
      candidate = bkg->getRandomPoint();
      uniform = log(R::runif(0, 1));
      q = beta->link(bkg->getVarVec(candidate,
                                    marksPrime(i),
                                    marksExpected(x.rows() + i),
                                    marksShape, marksNugget, marksMu,
                                    INTENSITY_VARIABLES))(0);
      if (uniform > q) { // Assign to U
        storingCoords.row(totalPoints - ++accU) = candidate.transpose();
        bkg->acceptNewPoint(INTENSITY_VARIABLES);
        stillSampling = false;
      } else {
        p = delta->link(bkg->getVarVec(candidate,
                                       marksPrime(i),
                                       marksExpected(x.rows() + i),
                                       marksShape, marksNugget, marksMu,
                                       OBSERVABILITY_VARIABLES))(0);
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
  marksPrime = marksPrime.head(accXp);
  marksExpected = marksExpected.head(x.rows() + accXp);

  xprime = storingCoords.topRows(accXp);
  u = storingCoords.bottomRows(accU);

  xxprimeIntensity.resize(x.rows() + accXp, beta->getSize() - 1);
  xxprimeIntensity.topRows(x.rows()) = xIntensity;
  xprimeObservability.resize(accXp, delta->getSize() - 1);

  if (accXp) {
    xxprimeIntensity.bottomRows(accXp) =
      bkg->getVarMat(storingCoords.topRows(accXp), INTENSITY_VARIABLES);
    xprimeObservability.leftCols(delta->getSize() - 1) =
      bkg->getVarMat(storingCoords.topRows(accXp), OBSERVABILITY_VARIABLES);
    xprimeObservability.col(delta->getSize() - 1) =
      bkg->getGP(OBSERVABILITY_VARIABLES);
  }

  uIntensity.resize(accU, beta->getSize() - 1);
  if (accU)
    uIntensity = bkg->getVarMat(storingCoords.bottomRows(accU), INTENSITY_VARIABLES);

  return - lgamma(accXp + 1) - lgamma(accU + 1);
}

double PresenceOnly::updateMarksPars(const Eigen::VectorXd& gp) {
  double sqrtMarksNugget = sqrt(marksNugget);

  Eigen::VectorXd logExpected = marksExpected.array().log() - gp.array();

  // Sampling the nugget
  int counter = 0;
  double propNugget, propDens = 0, prevDens = 0, temp;
  do {
    propNugget = R::rnorm(marksNugget, 0.1);
  } while (propNugget < 0 && ++counter < 100);
  if (counter == 100)
    Rf_warning("Nugget parameter attempts reached max iterations without a positive value.");

  temp = (logExpected.array() - marksMu).matrix().squaredNorm();
  propDens = -0.5 * (propNugget + temp) - (marksNuggetPriora + 1) * log(marksNugget) - marksNuggetPriorb / marksNugget;
  prevDens = -0.5 * (marksNugget + temp) - (marksNuggetPriora + 1) * log(propNugget) - marksNuggetPriorb / propNugget;

  double partialDens = prevDens;
  if (log(R::runif(0, 1)) <= propDens - prevDens) {
    marksNugget = propNugget;
    partialDens = propDens;
  }

  // Sampling the mean parameter
  double newVariance = 1 / (1 / marksMuPriors2 + marksExpected.size() / marksNugget);
  marksMu = R::rnorm(
    newVariance *
      (marksMuPriormu / marksMuPriors2 + logExpected.sum() / marksNugget),
      sqrt(newVariance)
  );

  // Sampling the shape parameter
  counter = 0;
  double propShape;
  temp = marksExpected.array().log().sum() +
    (marks.array() / marksExpected.head(x.rows()).array()).sum() +
    (marksPrime.array() / marksExpected.tail(xprime.rows()).array()).sum();
  double logVals = marks.array().log().sum() + marksPrime.array().log().sum();
  prevDens = marksShape * (log(marksShape) - temp) - lgamma(marksShape) +
    (marksShape - 1) * logVals + (marksShapePriora - 1) * log(marksShape) -
    marksShapePriorb * marksShape;
  do {
    propShape = R::rnorm(marksShape, 0.1);
  } while (propShape < 0 && ++counter < 100);
  if (counter == 100) {
    Rf_warning("Shape parameter attempts reached max iterations without a positive value.");
  } else {
    propDens = propShape * (log(propShape) - temp) - lgamma(propShape) +
      (propShape - 1) * logVals + (marksShapePriora - 1) * log(propShape) -
      marksShapePriorb * propShape;
    if (log(R::runif(0, 1)) <= propDens - prevDens) {
      marksShape = propShape;
      return partialDens + propDens;
    }
  }
  return partialDens + prevDens;
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
  out += updateMarksPars(bkg->getGPfull(OBSERVABILITY_VARIABLES));
  bkg->resampleGPs(marksMu, marksNugget, marksShape, marksExpected,
                 marks, marksPrime, delta->getNormalMean(), delta->getExtra());
  return out + privateOut1 + privateOut2;
}

