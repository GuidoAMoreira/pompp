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

#endif
