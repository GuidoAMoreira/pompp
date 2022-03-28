#include "include/BinaryRegression.hpp"
#include <RcppEigen.h>
#include "include/PolyaGamma.h"
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

// Algorithm described in Polson et al. (2012)
double LogisticRegression::sample(const Eigen::MatrixXd& onesCovariates,
                                  const Eigen::MatrixXd& zerosCovariates) {
  unsigned long i, n1 = onesCovariates.rows(), n0 = zerosCovariates.rows();
  pg = std::vector<double>(n0 + n1);
  PolyaGamma PG(1);
  setNormalMean(Eigen::Vector(n0 + n1));
  Eigen::MatrixXd V = Eigen::MatrixXd::Constant(n, n, 0),
    x1 = Eigen::MatrixXd(n1, n),
    x0 = Eigen::MatrixXd(n0, n);
  Eigen::VectorXd med = Eigen::MatrixXd::Constant(n, 1, 0), xb1(n1), xb0(n0);
  x1.leftCols(1) = Eigen::MatrixXd::Constant(n1,1,1);
  x1.rightCols(n - 1) = onesCovariates;
  x0.leftCols(1) = Eigen::MatrixXd::Constant(n0,1,1);
  x0.rightCols(n - 1) = zerosCovariates;
  xb1 = x1 * betas;
  xb0 = x0 * betas;

  // Calculating X' Omega X + B and X' kappa + B b
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
  Eigen::VectorXd priMed = Eigen::MatrixXd::Constant(n, 1, 0);
  Eigen::MatrixXd priV = Eigen::MatrixXd::Constant(n, n, 0);
#ifdef _OPENMP
#pragma omp for nowait
#endif
  for (i = 0; i < n1; i++) // From the data matrix X
  {
    pg[i] = PG.draw_like_devroye(xb1(i));
    priV += pg[i] * x1.row(i).transpose() * x1.row(i);
    priMed += x1.row(i) * 0.5;
    setNormalMeanIndex(0.5 - (xb1(i) - x1(i, n - 1) * betas(n)) * pg[i], i);
  }
#ifdef _OPENMP
#pragma omp for nowait
#endif
  for (i = 0; i < n0; i++) // From the data matrix X
  {
    pg[n1 + i] = PG.draw_like_devroye(xb0(i));
    priV += pg[n1 + i] * x0.row(i).transpose() * x0.row(i);
    priMed -= x0.row(i) * 0.5;
    setNormalMeanIndex(-0.5 - (xb0(i) - x0(i, n - 1) * betas(n)) * pg[n1 + i], i);
  }
#ifdef _OPENMP
#pragma omp critical
#endif
{
  V += priV;
  med += priMed;
}
  }

  betas = prior->sample(med, V);

  return link(x1, betas, false).sum() + link(x0, betas, true).sum() +
    prior->logPrior(betas);
}

// Logistic link in the log scale
inline Eigen::VectorXd LogisticRegression::link(const Eigen::MatrixXd& covariates,
                                                const Eigen::VectorXd& beta,
                                                bool complementaryProb) {
  return -( ( (complementaryProb ? 1 : -1) * (beta(0) +
    (covariates * beta.tail<n>()).array() ) ).exp().log1p());
}

