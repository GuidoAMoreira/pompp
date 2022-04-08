#ifndef __PRESENCE_ONLY_HPP__
#define __PRESENCE_ONLY_HPP__

#include "MarkovChain.hpp"
#include "BinaryRegression.hpp"
#include "BackgroundVariables.hpp"

#define MAX_ATTEMPTS_GP 1000

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
  BackgroundVariables* bkg;

  // lambda star members
  double lambdaStar;
  double aL, bL;
  double updateLambdaStar();

  // Marks members
  const Eigen::VectorXd marks;
  const double marksShapePriora, marksShapePriorb;
  Eigen::VectorXd marksPrime, marksExpected;
  double marksMu, marksShape, marksNugget, previousMarksLogDens;
  double updateMarks(const Eigen::VectorXd& gp);

public:
  PresenceOnly(const Eigen::MatrixXd& xPositions,
               const Eigen::MatrixXd& xIntensityCovs,
               const Eigen::MatrixXd& xObservabilityCovs,
               BackgroundVariables* bk,
               const Eigen::VectorXd& observedValues,
               double a, double mpa, double mpb) : area(a), x(xPositions),
               xIntensity(xIntensityCovs),
               xObservability(xObservabilityCovs), bkg(bk),
               marks(observedValues), marksShapePriora(mpa),
               marksShapePriorb(mpb), previousMarksLogDens(-INFINITY) {}

protected:
  double applyTransitionKernel();
};

#endif
