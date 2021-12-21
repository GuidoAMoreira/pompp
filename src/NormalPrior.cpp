#include <RcppEigen.h>
#include "include/NormalPrior.hpp"

Eigen::VectorXd NormalPrior::sample(const Eigen::VectorXd& mean,
                                    const Eigen::MatrixXd& precision) {
  Eigen::MatrixXd V = (precision + priorPrecision).inverse();
  Eigen::VectorXd m = V * (mean + precisionTimesMean);

  return V.llt().matrixL() *
    Rcpp::as<Eigen::Map<Eigen::VectorXd> >(Rcpp::rnorm(mean.size(), 0, 1)) + m;
}

inline double NormalPrior::logPrior(const Eigen::VectorXd& betas) {
  Eigen::VectorXd m = betas - priorMean;
  return m.transpose() * priorPrecision * m;
}
