#include <RcppEigen.h>
#include "include/RegressionPrior.hpp"

Eigen::VectorXd NormalPrior::sample(const Eigen::VectorXd& mean,
                                    const Eigen::MatrixXd& precision) {
  Eigen::LLT<Eigen::MatrixXd> decomp;
  decomp.compute(precision + priorPrecision);
  Eigen::MatrixXd temp = decomp.matrixL();

  return decomp.matrixL().transpose().solve(
                        Rcpp::as<Eigen::Map<Eigen::VectorXd> >(Rcpp::rnorm(mean.size(), 0, 1))
    ) +
    decomp.matrixL().transpose().solve(mean + precisionTimesMean);
}

inline double NormalPrior::logPrior(const Eigen::VectorXd& betas) {
  Eigen::VectorXd m = betas - priorMean;
  return m.transpose() * sigmaSolver.solve(m);
}
