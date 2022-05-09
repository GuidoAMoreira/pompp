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

void GaussianProcess::startUp(int howMany) {
  tempAcc = 0;
  augmentedPositions = positions;
  augmentedValues = values;
  augmentedCovariances = covariances;
  augmentedCovariances.reserve(howMany, howMany);
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
    augmentedCovariances.block(bigSize - tempAcc, 0, tempAcc, xSize);
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
  distances = Eigen::MatrixXd::Constant(augmentedPositions.rows(), 1, 0);
  std::fill(distances.data(), distances.data() + distances.size(), 0);
  neighborhood = getNeighorhood(coords);
  propPrecision = Eigen::MatrixXd(neighborhoodSize, neighborhoodSize);
  Eigen::VectorXd covariances(neighborhoodSize);
  double d;

#ifdef _OPENMP
#pragma omp parallel for private(j) collapse(2)
#endif
  for (int i = 0; i < neighborhoodSize; i++) {
    for (int j = 0; j < i; j++) {
      double* finder = std::find(
        pastCovariancesPositions.row(neighborhood[i]).data(),
        pastCovariancesPositions.row(neighborhood[i]).data() +
          pastCovariancesPositions.cols(),
        neighborhood[j]
      );
      propPrecision(i, j) =
        finder != pastCovariancesPositions.row(neighborhood[i]).data() +
        pastCovariancesPositions.cols() ?
          pastCovariances(neighborhood[i], std::distance(
              pastCovariancesPositions.row(neighborhood[i]).data(),
              finder
          ) - 1) :
        (*covFun)(calcDist(augmentedPositions.row(neighborhood[i]).transpose(),
                  augmentedPositions.row(neighborhood[j]).transpose()));
      propPrecision(j, i) = propPrecision(i, j);
    }
    propPrecision(i, i) = covFun->getSigma2();
    covariances(i) = (*covFun)(distances(neighborhood[i]));
    pastCovariancesPositions(tempAcc, i) = covariances(i);
  }
  Arow = propPrecision.llt().solve(covariances);
  D(tempAcc) = covFun->getSigma2() -
    (covariances.transpose() * Arow)(0);

  double t = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:t)
#endif
  for (int i = 0; i < neighborhoodSize; i++)
    t += augmentedValues(neighborhood[i]) * Arow(i);
  propValue = Rf_rnorm(t, sqrt(D(tempAcc)));
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

void NNGP::acceptNewPoint() {
  int n = augmentedValues.size();
  tempAcc++;
  for (int i = 0; i < neighborhoodSize; i++)
    trips.push_back(Eigen::Triplet<double>(tempAcc, i, -Arow(i)));
  trips.push_back(Eigen::Triplet<double>(tempAcc, tempAcc, 1.));
  augmentedValues.resize(n + 1);
  augmentedValues(n) = propValue;
}

void NNGP::startUp(int howMany) {
  tempAcc = xSize;
  augmentedPositions = positions;
  augmentedValues = values;
  trips = std::vector<Eigen::Triplet<double> >(neighborhoodSize * howMany);
  bootUpIminusA();
}

void NNGP::closeUp() {
  int newSize = xSize + tempAcc;
  int bigSize = augmentedCovariances.rows();
  positions.resize(newSize);
  positions.bottomRows(tempAcc) = augmentedPositions.bottomRows(tempAcc);
  values.resize(newSize);
  values.tail(tempAcc) = augmentedValues.tail(tempAcc);
  IminusA.setFromTriplets(trips.begin(), trips.end());
  IminusA.makeCompressed();
  precision = IminusA * D.asDiagonal() * IminusA.transpose();
}
