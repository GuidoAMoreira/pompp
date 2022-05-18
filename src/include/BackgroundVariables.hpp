#ifndef __BACKGROUND_VARIABLES_HPP__
#define __BACKGROUND_VARIABLES_HPP__

#include "GaussianProcess.hpp"
#include <RcppEigen.h>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

#define INTENSITY_VARIABLES 0
#define OBSERVABILITY_VARIABLES 1

class BackgroundVariables {
  const std::vector<int> intensityCols, observabilityCols;
  GaussianProcess* spatialProcessInt;
  GaussianProcess* spatialProcessObs;
  bool useGPint = false, useGPobs = false;

  Eigen::MatrixXd getVariablesMat(const Eigen::MatrixXd& coordinates,
                               std::vector<int> columns) {
    int n = coordinates.rows();
    Eigen::MatrixXd out(n, columns.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++)
      out.row(i) = getVariablesVec(coordinates.row(i), columns);

    return out;
  }
public:
  // Getters
  Eigen::MatrixXd getVarMat(const Eigen::MatrixXd& coordinates, int type) {
    if (type == INTENSITY_VARIABLES) return getVariablesMat(coordinates, intensityCols);
    if (type == OBSERVABILITY_VARIABLES) return getVariablesMat(coordinates, observabilityCols);
    return Eigen::MatrixXd(0, 0);
  }
  Eigen::VectorXd getVarVec(const Eigen::VectorXd& coordinates,
                            double& mark, double& markExpected,
                            double shape, double nugget, double mu, int type) {
    if (type == INTENSITY_VARIABLES) {
      Eigen::VectorXd out(intensityCols.size() + (useGPint ? 1 : 0));
      out.head(intensityCols.size()) = getVariablesVec(coordinates, intensityCols);
      if (useGPint)
        out(intensityCols.size()) =
          spatialProcessInt->getNewPoint(coordinates.head(2),
            mark, markExpected, shape, nugget, mu);
      return out;
    }
    if (type == OBSERVABILITY_VARIABLES) {
      Eigen::VectorXd out(observabilityCols.size() + (useGPobs ? 1 : 0));
      out.head(observabilityCols.size()) = getVariablesVec(coordinates, observabilityCols);
      if (useGPobs)
        out(observabilityCols.size()) =
          spatialProcessObs->getNewPoint(coordinates.head(2),
            mark, markExpected, shape, nugget, mu);
      return out;
    }
    return Eigen::VectorXd(0);
  }
  Eigen::VectorXd getGP(int type) {
    if (type == INTENSITY_VARIABLES) return spatialProcessInt->getAugmentedValuesTail();
    if (type == OBSERVABILITY_VARIABLES) return spatialProcessObs->getAugmentedValuesTail();
  }
  // Setters
  void setGP(GaussianProcess* gp, int type) {
    if (type == INTENSITY_VARIABLES) {
      spatialProcessInt = gp;
      useGPint = true;
      return;
    }
    if (type == OBSERVABILITY_VARIABLES) {
      spatialProcessObs = gp;
      useGPobs = true;
    }
  }

  // Resampler
  void resampleGPs(double marksMu, double marksVariance,
                   double marksShape, Eigen::VectorXd& marksExpected,
                   const Eigen::VectorXd& xMarks, Eigen::VectorXd& xPrimeMarks,
                   Eigen::VectorXd betasPart, Eigen::VectorXd pgs) {
    if (useGPint) spatialProcessInt->resampleGP(marksMu, marksVariance,
        marksShape, marksExpected, xMarks, xPrimeMarks, betasPart, pgs);
    if (useGPobs) spatialProcessObs->resampleGP(marksMu, marksVariance,
        marksShape, marksExpected, xMarks, xPrimeMarks, betasPart, pgs);
  }

  // Random point (for point process simulation)
  virtual Eigen::VectorXd getRandomPoint() = 0;
  Eigen::MatrixXd getRandomPoint(int size) {
    Eigen::MatrixXd output(size, 2);
    for (int i = 0; i < size; i++)
      output.row(i) = getRandomPoint().transpose();
    return output;
  }

  // Related to spatial Gaussian Processes
  void startGPs(int howMany) {
    if (useGPint) spatialProcessInt->startUp(howMany);
    if (useGPobs) spatialProcessObs->startUp(howMany);
  }
  void acceptNewPoint(int type) {
    if (type == INTENSITY_VARIABLES && useGPint)
      spatialProcessInt->acceptNewPoint();
    if (type == OBSERVABILITY_VARIABLES && useGPobs)
      spatialProcessObs->acceptNewPoint();
  }
  void setGPinStone() {
    if (useGPint) spatialProcessInt->closeUp();
    if (useGPobs) spatialProcessObs->closeUp();
  }

  // Constructor and destructor
  BackgroundVariables(std::vector<int> intCols, std::vector<int> obsCols,
                      GaussianProcess* gp) :
    intensityCols(intCols), observabilityCols(obsCols),
    spatialProcessObs(gp), useGPobs(true) {}
  virtual ~BackgroundVariables() {}

protected:
  std::vector<double*> data;

  // Retrieve variables for any set of columns
  virtual Eigen::VectorXd getVariablesVec(const Eigen::VectorXd& coordinates,
                                       std::vector<int> columns) = 0;
};

class MatrixVariables : public BackgroundVariables {
  const long rows, cols, longCol, latCol;
public:
  MatrixVariables(std::vector<int> intCols, std::vector<int> obsCols,
                  SEXP matrix, int xC, int yC, GaussianProcess* gp) :
                  BackgroundVariables(intCols, obsCols, gp),
                  rows(INTEGER(Rf_getAttrib(matrix, R_DimSymbol))[0]),
                  cols(INTEGER(Rf_getAttrib(matrix, R_DimSymbol))[1]),
                  longCol(xC), latCol(yC) {
    data = std::vector<double*>(cols);
    for (int i = 0; i < cols; i++) {
      data[i] = &REAL(matrix)[i * rows];
    }
  }

  // First coordinate is the random row and second is irrelevant.
  Eigen::VectorXd getRandomPoint() {
    Eigen::VectorXd out(3);
    long selectedRow = long(R::runif(0, 1) * rows);
    out(0) = data[longCol][selectedRow];
    out(1) = data[latCol][selectedRow];
    out(2) = selectedRow;
    return out;
  }

  // First coordinate is the desired row and second is irrelevant.
  Eigen::VectorXd getVariablesVec(const Eigen::VectorXd& coordinates,
                               std::vector<int> columns) {
    Eigen::VectorXd out(columns.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < columns.size(); j++)
      out(j) = data[columns[j]][(int)coordinates(2)];
    return out;
  }
};

#endif
