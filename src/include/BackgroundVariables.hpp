#ifndef __BACKGROUND_VARIABLES_HPP__
#define __BACKGROUND_VARIABLES_HPP__

#include <RcppEigen.h>

class BackgroundVariables {
  const std::vector<int> intensityCols, observabilityCols;
public:
  // Getters
  Eigen::MatrixXd getIntensityVar(const Eigen::MatrixXd& coordinates) {
    getVariables(coordinates, intensityCols);
  }
  Eigen::MatrixXd getObservabilityVar(const Eigen::MatrixXd& coordinates) {
    getVariables(coordinates, observabilityCols);
  }

  // Random point (for point process simulation)
  virtual Eigen::VectorXd getRandomPoint() = 0;
  Eigen::MatrixXd getRandomPoint(int size) {
    Eigen::MatrixXd output(size, 2);
    for (int i = 0; i < size; i++)
      output.row(i) = getRandomPoint().transpose();
    return output;
  }

  // Constructor
  BackgroundVariables(std::vector<int> intCols, std::vector<int> obsCols) :
    intensityCols(intCols), observabilityCols(obsCols) {}

protected:
  std::vector<double*> data;

  // Retrieve variables for any set of columns
  virtual Eigen::MatrixXd getVariables(const Eigen::MatrixXd& coordinates,
                                       std::vector<int> columns) = 0;
};

#endif
