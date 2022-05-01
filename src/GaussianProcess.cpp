#include "include/GaussianProcess.hpp"
#include <RcppEigen.h>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

void GaussianProcess::sampleNewPoint(Eigen::VectorXd coords) {
  Eigen::LLT<Eigen::MatrixXd> solver;
  Eigen::VectorXd temp;
  propCovariances = Eigen::VectorXd(augmentedPositions.rows());

#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int i = 0; i < augmentedPositions.rows(); i++) {
  propCovariances(i) = (*covFun)(calcDist(augmentedPositions.row(i).transpose(),
                                       coords));
}
  solver.compute(augmentedCovariances);
  temp = solver.solve(propCovariances);

  propValue = R::rnorm(temp.transpose() * augmentedValues,
                       covFun->getSigma2() - propCovariances.transpose() * temp);
}

double GaussianProcess::updateCovarianceParameters() {
  int estimSize = covFun->getParSize();
  std::vector<double> props(estimSize);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < estimSize; i++) {
    int attempts = 0;
    do {
      props[i] = R::rnorm(covFun->getPar(i), 0.1);
    } while (props[i] <= 0 && ++attempts <= 100);
    if (attempts == 100) {
      Rf_warning("Covariance parameter attempts reached max attempts.");
      props[i] = covFun->getPar(i);
    }
  }

  Eigen::MatrixXd propPrec = recalcPrecision(props);
  double newDensity = -0.5 * (values.transpose() * propPrec * values - log(propPrec.determinant()));
  if (log(R::runif(0, 1)) <= newDensity - logDensity) {
    for (int i = 0; i < estimSize; i++) covFun->setPar(props[i], i);
    return newDensity;
  } else return logDensity;
}

Eigen::MatrixXd GaussianProcess::recalcPrecision(std::vector<double> newParams) {
  return Eigen::MatrixXd(0, 0);
}

void GaussianProcess::acceptNewPoint() {
  int n = augmentedValues.size();
  augmentedValues.resize(n + 1);
  augmentedValues(n) = propValue;
  augmentedCovariances.resize(n + 1, n + 1);
  augmentedCovariances.row(n).head(n) = propCovariances;
  augmentedCovariances.col(n).head(n) = propCovariances;
  augmentedCovariances(n, n) = covFun->getSigma2();
  tempAcc++;
}

void GaussianProcess::startUp() {
  tempAcc = 0;
  augmentedPositions = positions;
  augmentedValues = values;
  augmentedCovariances = covariances;
}

void GaussianProcess::closeUp() {
  int newSize = xSize + tempAcc;
  int bigSize = augmentedCovariances.rows();
  positions.resize(newSize);
  positions.bottomRows(tempAcc) = augmentedPositions.bottomRows(tempAcc);
  values.resize(newSize);
  values.tail(tempAcc) = augmentedValues.tail(tempAcc);
  covariances.resize(newSize, newSize);
  covariances.block(tempAcc, tempAcc, tempAcc, tempAcc) =
    augmentedCovariances.block(bigSize - tempAcc, bigSize - tempAcc, tempAcc, tempAcc);
  covariances.block(xSize, 0, tempAcc, xSize) =
    augmentedCovariances.block(0, bigSize - tempAcc, xSize, tempAcc);
  covariances.block(0, xSize, xSize, tempAcc) =
    augmentedCovariances.block(bigSize - tempAcc, 0, tempAcc, xSize);index
}

double GaussianProcess::calcDist(Eigen::VectorXd p1, Eigen::VectorXd p2) {
  double temp, d = 0;
  for (int i = 0; i < 2; i++) {
    temp = p1(i) - p2(i);
    d += temp * temp;
  }
  return sqrt(d);
}

void NNGP::sampleNewPoint(Eigen::VectorXd coords) {
  distances = Eigen::VectorXd(augmentedPositions.rows());
  std::fill(distances.begin(), distances.end(), 0);
  std::vector<int> neighborhood = getNeighorhood(coords);
  Eigen::MatrixXd conditionalCovariance(neighborhoodSize, neighborhoodSize), temp;
  Eigen::VectorXd covariances(neighborhoodSize);
  Eigen::RowVectorXd::InnerIterator finder;
  double d;
  thisPosition++;

#ifdef _OPENMP
#pragma omp parallel for private(j, finder) collapse(2)
#endif
  for (int i = 0; i < neighborhoodSize; i++) {
    for (int j = 0; j < i; j++) {
      finder = std::find(
        pastCovariancesPositions.row(neighborhood[i]).begin(),
        pastCovariancesPositions.row(neighborhood[i]).end(),
        neighborhood[j]
      );
      conditionalCovariance(i, j) =
        finder != pastCovariancesPositions.row(neighborhood[i]).end() ?
          pastCovariances(neighborhood[i], std::distance(
              pastCovariancesPositions.row(neighborhood[i]).begin(),
              finder
          ) - 1) :
        (*covFun)(calcDist(augmentedPositions.row(neighborhood[i]).transpose(),
                  augmentedPositions.row(neighborhood[j]).transpose()));
      conditionalCovariance(j, i) = conditionalCovariance(i, j);
    }
    conditionalCovariance(i, i) = covFun->getSigma2();
    covariances(i) = (*covFun)(distances(neighborhood(i)));
    pastCovariancesPositions(thisPosition, i) = covariances(i);
  }
  temp = conditionalCovariance.llt().solve(covariances);
  D(thisPosition) = covFun->getSigma2() -
    (covariances.transpose() * temp)(0);

  double t = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:t)
#endif
  for (int i = 0; i < neighborhoodSize; i++)
    t += augmentedValues(neighborhood[i]) * temp(i);
  augmentedValues(thisPosition) = Rf_rnorm(t, sqrt(D(thisPosition)));
}

std::vector<int> NNGP::getNeighorhood(Eigen::VectorXd coords) {
  std::vector<int> output(augmentedPositions.rows());

  std::iota(output.begin(), output.end(), 0);
  std::nth_element(output.begin(), output.begin() + neighborhoodSize - 1, output.end(),
                   [this, &coords](int i1, int i2) {
                     if (!this->distances(i1))
                       this->distances(i1) = this->calcDist(coords, this->augmentedPositions.row(i1).transpose());
                     if (!this->distances(i2))
                       this->distances(i2) = this->calcDist(coords, this->augmentedPositions.row(i2).transpose());
                     return this->distances(i1) < this->distances(i2);
                     });

  return std::vector<int>(output.begin(), output.begin() + neighborhoodSize - 1);
}
