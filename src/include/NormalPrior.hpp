#ifndef __POMPP_NORMAL_PRIOR_HPP__
#define __POMPP_NORMAL_PRIOR_HPP__

#include <RcppEigen.h>
#include "RegressionPrior.hpp"

class NormalPrior : public RegressionPrior {
  const Eigen::VectorXd priorMean, precisionTimesMean;
  const Eigen::MatrixXd priorCovariance, priorPrecision;

public:
  NormalPrior(const Eigen::VectorXd& mu,
              const Eigen::VectorXd& Sigma) :
  priorMean(mu), precisionTimesMean(Sigma.solve(mu)),
  priorCovariance(Sigma), priorPrecision(Sigma.inverse()) {}

  Eigen::VectorXd sample(const Eigen::VectorXd& mean,
                         const Eigen::MatrixXd& precision);
  double logPrior(const Eigen::VectorXd& betas);

  // Some getters
  Eigen::VectorXd getMean() {return priorMean;}
  Eigen::MatrixXd getCovariance() {return priorCovariance;}
  Eigen::MatrixXd getPrecision() {return priorPrecision;}
};

#endif
