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
    bkg->startGPs(0);
    return 0.;
  }

  // Sampling from X' and U
  totalPoints -= x.rows(); // Number of points associated to them
  double p, q, uniform;
  long accXp = 0, accU = 0, currentAttempt;
  Eigen::VectorXd candidate;
  Eigen::MatrixXd storingCoords(totalPoints, 2); // Put X' on top and U on bottom
  bkg->startGPs(totalPoints);
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
      bkg->getVarMat(storingCoords.topRows(accXp), INTENSITY_VARIABLES);
    xprimeObservability = bkg->getVarMat(storingCoords.topRows(accXp), OBSERVABILITY_VARIABLES);
  }

  uIntensity.resize(accU, beta->getSize() - 1);
  if (accU)
    uIntensity = bkg->getVarMat(storingCoords.bottomRows(accU), INTENSITY_VARIABLES);

  return - lgamma(accXp + 1) - lgamma(accU + 1);
}

double PresenceOnly::updateMarks(const Eigen::VectorXd& gp) {
  double sqrtMarksNugget = sqrt(marksNugget);

  marksExpected = Eigen::VectorXd(gp.size());
  marksPrime = Eigen::VectorXd(xprime.rows());

  // Sampling the nugget
  int counter = 0;
  double propNugget, propDens = 0, prevDens = 0, temp, logEta;
  do {
    propNugget = R::rnorm(marksNugget, 0.1);
  } while (propNugget < 0 && ++counter < 100);
  if (counter == 100)
    Rf_warning("Nugget parameter attempts reached max iterations without a positive value.");
#ifdef _OPENMP
#pragma omp parallel for private(temp, logEta) reduction(+:propDens, prevDens)
#endif
  for (int i = 0; i < marks.size(); i++) {
    logEta = metropolisExpected(marks(i), gp(i), sqrtMarksNugget);
    marksExpected(i) = exp(logEta);
    temp = pow(logEta - marksMu - gp(i), 2);
    propDens += -0.5 / propNugget * temp;
    prevDens += -0.5 / marksNugget * temp;
  }
#ifdef _OPENMP
#pragma omp parallel for private(temp, logEta) reduction(+:propDens, prevDens)
#endif
  for (int i = marks.size(); i < gp.size(); i++) {
    logEta = R::rnorm(marksMu, sqrtMarksNugget) + gp(i);
    marksExpected(i) = exp(logEta);
    marksPrime(i - marks.size()) = R::rgamma(marksShape, marksShape / marksExpected(i));
    temp = pow(logEta - marksMu - gp(i), 2);
    propDens += -0.5 / propNugget * temp;
    prevDens += -0.5 / marksNugget * temp;
  }
  prevDens += -(marksNuggetPriora + 1) * log(marksNugget) - marksNuggetPriorb / marksNugget;
  propDens += -(marksNuggetPriora + 1) * log(propNugget) - marksNuggetPriorb / propNugget;
  double partialDens = prevDens;
  if (log(R::runif(0, 1)) <= propDens - prevDens) {
    marksNugget = propNugget;
    partialDens = propDens;
  }

  // Sampling the mean parameter
  marksMu = R::rnorm(
    1 / (1 / marksMuPriors2 + 1 / )
  );

  // Sampling the shape parameter
  counter = 0;
  double propShape;
  double temp = marksExpected.array().log().sum() +
    (marks.array() ./ marksExpected.head(xprime.rows()).array()).sum() +
    (marksPrime.array() ./ marksExpected.tail(gp.size() - marks.size()).array()).sum();
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

double PresenceOnly::metropolisExpected(double mark, double gpv, double sd) {
  double prop;
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
  out += updateMarks(bkg->getGP(OBSERVABILITY_VARIABLES));
  bkg->resampleGPs(marksMu, marksExpected,
                 marksNugget, beta->getNormalMean(), beta->getExtra());
  return out + privateOut1 + privateOut2;
}
