#include "include/GaussianProcess.hpp"
#include <RcppEigen.h>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

void GaussianProcess::sampleNewPoint(Eigen::VectorXd coords, double& mark,
                                     double& markExpected,
                                     double shape, double nugget, double mu) {
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
  augmentedCovariances = Eigen::MatrixXd(howMany, howMany);
  augmentedCovariances.block(0, 0, xSize, xSize) = covariances;
}

void GaussianProcess::closeUp() {
  int newSize = xSize + tempAcc;
  int bigSize = augmentedCovariances.rows();
  positions.resize(newSize, 2);
  positions.bottomRows(tempAcc) = augmentedPositions.bottomRows(tempAcc);
  values.resize(newSize);
  values.tail(tempAcc) = augmentedValues.tail(tempAcc);
  covariances.conservativeResize(newSize, newSize);
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

void GaussianProcess::resampleGP(double marksMu, double marksVariance,
                                 double marksShape, Eigen::VectorXd& marksExpected,
                                 const Eigen::VectorXd& xMarks, Eigen::VectorXd& xPrimeMarks,
                                 Eigen::VectorXd betasPart, Eigen::VectorXd pgs) {
  int n = marksExpected.size();
  Eigen::VectorXd temp = Eigen::MatrixXd::Constant(n, 1, 1 / marksVariance);
  Eigen::MatrixXd newPrec = covariances.inverse() + pgs + temp;

  augmentedValues = newPrec.llt().matrixL().transpose().solve(
    Rcpp::as<Eigen::Map<Eigen::VectorXd> >(Rcpp::rnorm(n, 0, 1))
  ) + betasPart +( (marksExpected.array().log() - marksMu) / marksVariance).matrix();
}

//// NNGP starts here ////

void NNGP::sampleNewPoint(Eigen::VectorXd coords, double& mark,
                          double& markExpected,
                          double shape, double nugget, double mu) {
  R_CheckUserInterrupt();

  propPosition = coords;
  distances = Eigen::MatrixXd::Constant(tempAcc, 1, 0); // Filled in getNeighborhood function. Must be 0.
  neighborhood = getNeighorhood(coords);
  propPrecision = Eigen::MatrixXd(neighborhoodSize, neighborhoodSize);
  theseCovariances = Eigen::VectorXd(neighborhoodSize);
  int finder;

#ifdef _OPENMP
#pragma omp parallel for private(finder)
#endif
  for (int i = 0; i < neighborhoodSize; i++) {
    for (int j = 0; j < i; j++) {
      for (finder = 0; finder < neighborhoodSize; finder++)
        if (pastCovariancesPositions(neighborhood[i], finder) ==
            neighborhood[j]) break;
        propPrecision(i, j) =
          finder < neighborhoodSize ?
          pastCovariances(neighborhood[i], finder) :
          (*covFun)(calcDist(augmentedPositions.row(neighborhood[i]).transpose(),
                    augmentedPositions.row(neighborhood[j]).transpose()));
        propPrecision(j, i) = propPrecision(i, j);
    }
    propPrecision(i, i) = covFun->getSigma2();
    theseCovariances(i) = (*covFun)(distances(neighborhood[i]));
  }
  Arow = propPrecision.llt().solve(theseCovariances);
  propD = covFun->getSigma2() - (theseCovariances.transpose() * Arow)(0);

  double iterationMean = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:iterationMean)
#endif
  for (int i = 0; i < neighborhoodSize; i++)
    iterationMean += augmentedValues(neighborhood[i]) * Arow(i);

  // Variable transformation
  double x, logy, y, z, totalVariance = propD + nugget;
  x = R::rgamma(shape, 1 / shape);
  logy = R::rnorm(mu + iterationMean + 2 * (totalVariance), sqrt(totalVariance));
  y = exp(logy);
  z = R::rnorm(iterationMean * nugget / (totalVariance), sqrt(propD * nugget) /
    totalVariance);

  markExpected = y;
  mark = x * y;
  propValue = z * totalVariance + propD * (logy - mu);
}

std::vector<int> NNGP::getNeighorhood(Eigen::VectorXd coords) {
  std::vector<int> output(tempAcc);

  std::iota(output.begin(), output.end(), 0);
  std::nth_element(output.begin(), output.begin() + neighborhoodSize - 1, output.end(),
                   [this, &coords](int i1, int i2) {
                     if (!this->distances(i1))
                       this->distances(i1) = this->calcDist(coords, this->augmentedPositions.row(i1).transpose());
                     if (!this->distances(i2))
                       this->distances(i2) = this->calcDist(coords, this->augmentedPositions.row(i2).transpose());
                     return this->distances(i1) < this->distances(i2);
                   });

  return std::vector<int>(output.begin(), output.begin() + neighborhoodSize);
}

void NNGP::acceptNewPoint() {
  int i;
  augmentedValues(tempAcc) = propValue;
  augmentedPositions.row(tempAcc) = propPosition.transpose();
  Eigen::VectorXi acceptedPositions(neighborhoodSize);
  for (i = 0; i < neighborhoodSize; i++) acceptedPositions(i) = neighborhood[i];
  pastCovariancesPositions.row(tempAcc) = acceptedPositions.transpose();
  pastCovariances.row(tempAcc) = theseCovariances.transpose();
  D(tempAcc) = 1 / propD;
  for (i = 0; i < neighborhoodSize; i++)
    trips.push_back(Eigen::Triplet<double>(tempAcc, neighborhood[i], -Arow(i)));
  trips.push_back(Eigen::Triplet<double>(tempAcc, tempAcc, 1.));
  tempAcc++;
}

void NNGP::startUp(int howMany) {
  tempAcc = neighborhoodSize;
  augmentedPositions.resize(xSize + howMany, 2);
  augmentedPositions.topRows(xSize) = positions;
  augmentedValues.resize(xSize + howMany);
  augmentedValues.head(xSize) = values;
  pastCovariancesPositions.resize(xSize + howMany, neighborhoodSize);
  pastCovariances.resize(xSize + howMany, neighborhoodSize);
  trips = std::vector<Eigen::Triplet<double> >(xSize * xSize + neighborhoodSize * howMany);
  D.resize(xSize + howMany);
  bootUpIminusA();
}

void NNGP::closeUp() {
  IminusA = Eigen::SparseMatrix<double>(tempAcc, tempAcc);
  IminusA.setFromTriplets(trips.begin(), trips.end());
  IminusA.makeCompressed();
  D.conservativeResize(tempAcc);
  precision = IminusA * D.asDiagonal() * IminusA.transpose();
  augmentedValues.conservativeResize(tempAcc);
  // Close up the extra spaces
  trips = std::vector<Eigen::Triplet<double> >(0);
}

void NNGP::resampleGP(double marksMu, double marksVariance,
                      double marksShape, Eigen::VectorXd& marksExpected,
                      const Eigen::VectorXd& xMarks, Eigen::VectorXd& xPrimeMarks,
                      Eigen::VectorXd betasPart, Eigen::VectorXd pgs) {
  int n = marksExpected.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < xSize; i++) {
    marksExpected(i) = 1 / R::rgamma(marksShape, 1 / (xMarks(i) * marksShape));
  }

  precision.diagonal().array() += pgs.array() + 1 / marksVariance;
  sqrtC.compute(precision);
  Eigen::VectorXd temp = sqrtC.matrixU().solve(
    Rcpp::as<Eigen::Map<Eigen::VectorXd> >(Rcpp::rnorm(n, 0, 1))
  );
  augmentedValues = temp + betasPart + ((marksExpected.array().log() - marksMu) / marksVariance).matrix();
  values = augmentedValues.head(xSize);

  Eigen::MatrixXd printer(marksExpected.size(), 6);
  printer.col(0) = marksExpected;
  printer.col(1) = augmentedValues;
  printer.col(2) = temp;
  printer.col(3) = betasPart;
  printer.col(4) = ((marksExpected.array().log() - marksMu) / marksVariance).matrix();
  printer.col(5) = ((marksExpected.array().log() - marksMu)).matrix();
  Rcpp::Rcout << marksMu << std::endl;
  Rcpp::Rcout << printer << std::endl << std::endl;
}

void NNGP::bootUpIminusA() {
  int i, j, k;
  Eigen::VectorXd temp;
  covariances = Eigen::MatrixXd(neighborhoodSize, neighborhoodSize);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < neighborhoodSize; i++) {
    for (int j = 0; j < i; j++) {
      covariances(i, j) = (*covFun)(calcDist(positions.row(i).transpose(),
                           positions.row(j).transpose()));
      covariances(j, i) = covariances(i, j);
    }
    covariances(i, i) = covFun->getSigma2();
  }

  Eigen::LDLT<Eigen::MatrixXd> miniSolver;
  miniSolver.compute(covariances);
  Eigen::MatrixXd miniIminusA = miniSolver.matrixL().solve(Eigen::MatrixXd::Identity(xSize, xSize));
  for (i = 0; i < neighborhoodSize; i++) {
    for (j = 0; j < i; j++)
      trips.push_back(Eigen::Triplet<double>(i, j, miniIminusA(i, j)));
    trips.push_back(Eigen::Triplet<double>(i, i, 1.));
    D(i) = miniSolver.vectorD()(i);

    for (j = 0; j < neighborhoodSize; j++) {
      if (j < i) {
        pastCovariancesPositions(i, j) = j;
        pastCovariances(i, j) = covariances(i, j);
      } else {
        pastCovariancesPositions(i, j) = i;
        pastCovariances(i, j) = covFun->getSigma2();
      }
    }
  }
  for (k = neighborhoodSize; k < xSize; k++) {
    distances = Eigen::MatrixXd::Constant(k, 1, 0); // Filled in getNeighborhood function. Must be 0.
    neighborhood = getNeighorhood(positions.row(k));
    propPrecision = Eigen::MatrixXd(neighborhoodSize, neighborhoodSize);
    theseCovariances = Eigen::VectorXd(neighborhoodSize);
    for (i = 0; i < neighborhoodSize; i++) {
      for (j = 0; j < i; j++) {
        propPrecision(i, j) =
          (*covFun)(calcDist(positions.row(neighborhood[i]).transpose(),
                    positions.row(neighborhood[j]).transpose()));
        propPrecision(j, i) = propPrecision(i, j);
      }
      propPrecision(i, i) = covFun->getSigma2();
      theseCovariances(i) = (*covFun)(distances(neighborhood[i]));
    }
    Arow = propPrecision.llt().solve(theseCovariances);
    D(k) = 1 / (covFun->getSigma2() - (theseCovariances.transpose() * Arow)(0));

    Eigen::VectorXi acceptedPositions(neighborhoodSize);
    for (i = 0; i < neighborhoodSize; i++) acceptedPositions(i) = neighborhood[i];
    pastCovariancesPositions.row(k) = acceptedPositions.transpose();
    pastCovariances.row(k) = theseCovariances.transpose();
    for (i = 0; i < neighborhoodSize; i++)
      trips.push_back(Eigen::Triplet<double>(k, neighborhood[i], -Arow(i)));
    trips.push_back(Eigen::Triplet<double>(k, k, 1.));
    tempAcc++;
  }
}
