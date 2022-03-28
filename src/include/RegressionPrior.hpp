#ifndef __NORMAL_LIKELIHOOD_REGRESSION_PRIOR_HPP__
#define __NORMAL_LIKELIHOOD_REGRESSION_PRIOR_HPP__

#include <RcppEigen.h>

class RegressionPrior {
protected:
public:
  virtual Eigen::VectorXd sample(const Eigen::VectorXd& mean,
                                 const Eigen::MatrixXd& precision) = 0;
  virtual double logPrior(const Eigen::VectorXd& betas) = 0;
};

class NormalPrior : public RegressionPrior {
  const Eigen::MatrixXd priorCovariance;
  Eigen::MatrixXd priorPrecision;
  const Eigen::VectorXd priorMean;
  Eigen::VectorXd precisionTimesMean;
  Eigen::LLT<Eigen::MatrixXd> sigmaSolver;

public:
  NormalPrior(const Eigen::VectorXd& mu,
              const Eigen::VectorXd& Sigma) :
  priorMean(mu), priorCovariance(Sigma) {
    sigmaSolver.compute(Sigma);
    priorPrecision = sigmaSolver.solve(Eigen::Matrix::Identity(mu.size(), mu.size()));
    precisionTimesMean = sigmaSolver.solve(mu);
  }

  Eigen::VectorXd sample(const Eigen::VectorXd& mean,
                         const Eigen::MatrixXd& precision);
  double logPrior(const Eigen::VectorXd& betas);

  // Some getters
  Eigen::VectorXd getMean() {return priorMean;}
  Eigen::MatrixXd getCovariance() {return priorCovariance;}
};

#endif
