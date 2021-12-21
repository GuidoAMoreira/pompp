#ifndef __pompp_BINARY_REGRESSION_HPP__
#define __pompp_BINARY_REGRESSION_HPP__

#include <RcppEigen.h>
#include "RegressionPrior.hpp"

class BinaryRegression {
protected:
  Eigen::VectorXd betas;
  RegressionPrior* prior;
  const unsigned int n;

public:
  virtual double sample(const Eigen::MatrixXd& onesCovariates,
                        const Eigen::MatrixXd& zerosCovariates) = 0;


  // Link function. Returns in the **LOG** scale. complementaryProb = true
  // calculates the link for 1 - p
  // For the stored beta vector
  Eigen::VectorXd link(const Eigen::MatrixXd covariates,
                       bool complementaryProb = false) {
    return link(covariates, betas, complementaryProb);
  }

  // For any beta vector
  virtual Eigen::VectorXd link(const Eigen::MatrixXd covariates,
                               const Eigen::VectorXd beta,
                               bool complementaryProb = false) = 0;

  // Constructor
  BinaryRegression(Eigen::VectorXd initialize) : betas(initialize),
    n(initialize.size()) {}

  // Prior setter
  void setPrior(RegressionPrior* p) {prior = p;}

  // Some getters
  Eigen::VectorXd getBeta() {return betas;}
  int getSize() {return n;}
};

#endif
