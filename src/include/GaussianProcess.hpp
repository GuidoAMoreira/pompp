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
  virtual void acceptNewPoint();

  // Methods to update which points are data augmentation.
  virtual void startUp();
  virtual void closeUp();
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

  Eigen::VectorXd calcDist(Eigen::VectorXd p1, Eigen::VectorXd p2);
};

class NNGP : public GaussianProcess {

};

#endif
