#ifndef __LOGISTIC_REGRESSION_HPP__
#define __LOGISTIC_REGRESSION_HPP__

#include <RcppEigen.h>
#include "BinaryRegression.hpp"

class LogisticRegression : public BinaryRegression {
  double sample(const Eigen::MatrixXd& onesCovariates,
                const Eigen::MatrixXd& zerosCovariates);
  Eigen::VectorXd link(const Eigen::MatrixXd& covariates,
                       const Eigen::VectorXd& beta,
                       bool complementaryProb);

  LogisticRegression(Eigen::VectorXd initialize) : BinaryRegression(initialize) {}
};

#endif
