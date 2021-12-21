#ifndef __PRESENCE_ONLY_HPP__
#define __PRESENCE_ONLY_HPP__

#include "MarkovChain.hpp"
#include "BinaryRegression.hpp"
#include "BackgroundVariables.hpp"

class PresenceOnly : public MarkovChain {
  // States
  // Regression members
  BinaryRegression* beta;
  BinaryRegression* delta;

  // Point process members
  const double area;
  Eigen::MatrixXd xprime, u;
  Eigen::MatrixXd xxprimeIntensity, xprimeObservability, uIntensity;
  double sampleProcesses();

  // Data
  const Eigen::MatrixXd x;
  const Eigen::MatrixXd xIntensity, xObservability;
  const BackgroundVariables* bkg;

  // lambda star members
  double lambdaStar;
  double aL, bL;
  double updateLambdaStar();

public:
  PresenceOnly(const Eigen::MatrixXd& xPositions,
               const Eigen::MatrixXd& xIntensityCovs,
               const Eigen::MatrixXd& xObservabilityCovs,
               const BackgroundVariables* bk,
               double a) : area(a), x(xPositions), xIntensity(xIntensityCovs),
               xObservability(xObservabilityCovs), bkg(bk) {}

protected:
  double applyTransitionKernel();
};

#endif
