#ifndef __POMPP_GAUSSIANPROCESS_H__
#define __POMPP_GAUSSIANPROCESS_H__

#include <RcppEigen.h>
#include "CovarianceFunction.hpp"

class GaussianProcess {
  // coordinates parameters. Function returns the sampled value
  virtual void sampleNewPoint(Eigen::VectorXd coords);

  double updateCovarianceParameters();
  virtual Eigen::MatrixXd recalcPrecision(std::vector<double> newParams); // Used in updateCovarianceParameter()
public:
  // getters
  Eigen::VectorXd getValues() {return values;}
  // setters
  void setCovFunction(CovarianceFunction* c) {covFun = c;}

  GaussianProcess(int s) : xSize(s) {}
  GaussianProcess(Eigen::MatrixXd pos, int s) : xSize(s), tempSize(0), positions(pos) {}

  double getNewPoint(Eigen::VectorXd coords) {sampleNewPoint(coords); return propValue;}
  void acceptNewPoint();

  // Methods to update which points are data augmentation.
  void startUp();
  void closeUp();
protected:
  const int xSize; // Used in start up and close up
  int tempAcc, tempSize; // Used in start up and close up
  int parameterSize, currentIndex;
  Eigen::MatrixXd positions, covariances, precision, augmentedPositions, augmentedCovariances;
  Eigen::VectorXd values, augmentedValues;
  CovarianceFunction* covFun;
  double logDensity;

  // Proposed point
  Eigen::VectorXd propCovariances;
  double propValue;

  double calcDist(Eigen::VectorXd p1, Eigen::VectorXd p2);
};

class NNGP : public GaussianProcess {
  void sampleNewPoint(Eigen::VectorXd coords);
  Eigen::MatrixXd recalcPrecision(std::vector<double> newParams); // Used in updateCovarianceParameter()

  // Neighborhood members
  const int neighborhoodSize;
  Eigen::VectorXd distances, D;
  std::vector<int> getNeighorhood(Eigen::VectorXd coords);
  Eigen::SparseMatrix<double> IminusA;
  Eigen::LLT<Eigen::MatrixXd> sqrtC;
  Eigen::MatrixXd pastCovariancesPositions, pastCovariances;
  int thisPosition;

public:
  NNGP(Eigen::MatrixXd pos, int s, int M) : GaussianProcess(pos, s),
    neighborhoodSize(M) {}
};

#endif
