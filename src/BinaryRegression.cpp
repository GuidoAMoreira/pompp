#include "include/BinaryRegression.hpp"
extern "C" {
#include "include/PolyaGammaSampler.h"
}
#include <RcppEigen.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(openmp)]]

// Algorithm described in Polson et al. (2012)
double LogisticRegression::sample(const Eigen::MatrixXd& onesCovariates,
                                  const Eigen::MatrixXd& zerosCovariates) {
  unsigned long n1 = onesCovariates.rows(), n0 = zerosCovariates.rows();
  pg.resize(n0 + n1);
  Eigen::VectorXd newNormalMean(n0 + n1);
  Eigen::MatrixXd V = Eigen::MatrixXd::Constant(n, n, 0),
    x1 = Eigen::MatrixXd(n1, n),
    x0 = Eigen::MatrixXd(n0, n);
  Eigen::VectorXd med = Eigen::MatrixXd::Constant(n, 1, 0), xb1(n1), xb0(n0);
  x1.leftCols(1) = Eigen::MatrixXd::Constant(n1, 1, 1);
  x1.rightCols(n - 1) = onesCovariates;
  x0.leftCols(1) = Eigen::MatrixXd::Constant(n0, 1, 1);
  x0.rightCols(n - 1) = zerosCovariates;
  xb1 = x1 * betas;
  xb0 = x0 * betas;

  // Calculating X' Omega X + B and X' kappa + B b
#pragma omp parallel
{
  Eigen::VectorXd priMed = Eigen::MatrixXd::Constant(n, 1, 0);
  Eigen::MatrixXd priV = Eigen::MatrixXd::Constant(n, n, 0);
#pragma omp for nowait
  for (int i = 0; i < n1; i++) // From the data matrix X
  {
    pg[i] = draw_from_PolyaGamma(xb1(i));
    priV += pg[i] * x1.row(i).transpose() * x1.row(i);
    priMed += x1.row(i) * 0.5;
    newNormalMean(i) = (0.5 - (xb1(i) - x1(i, n - 1) * betas(n - 1)) * pg[i]) *
      betas(n - 1);
  }
#pragma omp for nowait
  for (int i = 0; i < n0; i++) // From the data matrix X
  {
    pg[n1 + i] = draw_from_PolyaGamma(xb0(i));
    priV += pg[n1 + i] * x0.row(i).transpose() * x0.row(i);
    priMed -= x0.row(i) * 0.5;
    newNormalMean(n1 + i) = (-0.5 - (xb0(i) - x0(i, n - 1) * betas(n - 1)) * pg[n1 + i]) *
      betas(n - 1);
  }
#pragma omp critical
{
  V += priV;
  med += priMed;
}
}
  setNormalMean(newNormalMean);

  betas = prior->sample(med, V);

  return link(onesCovariates, betas, false).sum() +
    link(zerosCovariates, betas, true).sum() +
    prior->logPrior(betas);
}

// Logistic link in the log scale
inline Eigen::VectorXd LogisticRegression::link(const Eigen::MatrixXd& covariates,
                                                const Eigen::VectorXd& beta,
                                                bool complementaryProb) {
  return -( ( (complementaryProb ? 1 : -1) * (beta(0) +
            (covariates * beta.tail(n - 1)).array() ) ).exp().log1p());
}
