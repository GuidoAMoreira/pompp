#include <RcppEigen.h>
#include "include/PresenceOnly.hpp"
#include "include/BackgroundVariables.hpp"

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
  long totalPoints = R::rpois(lambdaStar * area);
  if (totalPoints <= x.rows()) {
    xprime.resize(0, 2);
    u.resize(0, 2);
    xprimeIntensity.resize(0, beta->getSize() - 1);
    xprimeObservability.resize(0, delta->getSize() - 1);
    uIntensity.resize(0, beta->getSize() - 1);
    return 0.;
  }

  // Sampling from X' and U
  totalPoints -= x.rows(); // Number of points associated to them
  double p, q, u;
  long accXp = 0, accU = 0;
  Eigen::VectorXd candidate;
  Eigen::MatrixXd storingCoords(totalPoints, 2); // Put X' on top and U on bottom
  for (int i = 0; i < totalPoints; i++) {
    do {
      candidate = bkg->getRandomPoint();
      // TODO: Sample spatial effects
      q = beta->link(bkg->getIntensityVar(candidate.transpose()));
      p = delta->link(bkg->getObservabilityVar(candidate.transpose()));
      u = log(R::runif(0, 1));
    } while (u <= q + p);
    if (u > q) // assign to U
      storingCoords.row(totalPoints - ++accU) = candidate.transpose();
    else // assign to X'
      storingCoords.row(accXp++) = candidate.transpose();
  }

  xprime.resize(accXp, 2);
  u.resize(accU, 2);

  xxprimeIntensity.resize(x.rows() + accXp, beta->getSize() - 1);
  xprimeObservability.resize(accXp, delta->getSize() - 1);

  if (accXp) {
    xxprimeIntensity.bottomRows(accXp) =
      bkg->getIntensityVar(storingCoords.topRows(accXp));
    xprimeObservability = bkg->getObservabilityVar(storingCoords.topRows(accXp));
  }

  uIntensity.resize(accU, beta->getSize() - 1);
  if (accU)
    uIntensity = bkg->getIntensityVar(storingCoords.bottomRows(accU));

  return - lgamma(accXp + 1) - lgamma(accU + 1);
}

inline double PresenceOnly::applyTransitionKernel() {
  return sampleProcesses() +
    updateLambdaStar() +
    beta->sample(xxprimeIntensity, uIntensity) +
    delta->sample(xObservability, xprimeObservability);
}
