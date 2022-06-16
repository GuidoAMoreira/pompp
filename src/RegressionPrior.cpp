#include <RcppEigen.h>
#include "include/RegressionPrior.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "include/safeR.hpp"

Eigen::VectorXd NormalPrior::sample(const Eigen::VectorXd& mean,
                                    const Eigen::MatrixXd& precision) {
  Eigen::LLT<Eigen::MatrixXd> decomp;
  decomp.compute(precision + priorPrecision);
  Eigen::VectorXd output;

  output = decomp.matrixU().solve(rnorm(mean.size())) +
    decomp.solve(mean + precisionTimesMean);

  return output;
}

inline double NormalPrior::logPrior(const Eigen::VectorXd& betas) {
  Eigen::VectorXd m = betas - priorMean;
  return m.transpose() * sigmaSolver.solve(m);
}
