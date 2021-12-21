#ifndef __MATRIX_VARIABLES_HPP__
#define __MATRIX_VARIABLES_HPP__

#include <Rinternals.h>
#include "BackgroundVariables.hpp"

class MatrixVariables : public BackgroundVariables {
  const long rows, cols;
public:
  MatrixVariables(std::vector<int> intCols, std::vector<int> obsCols,
                  SEXP matrix) : BackgroundVariables(intCols, obsCols),
                  rows(INTEGER(Rf_getAttrib(matrix, R_DimSymbol))[0]),
                  cols(INTEGER(Rf_getAttrib(matrix, R_DimSymbol))[1]){
    data = std::vector<double*>(cols);
    for (int i = 0; i < cols; i++) {
      data[i] = &REAL(matrix)[i * rows];
    }
  }

  // First coordinate is the random row and second is irrelevant.
  Eigen::VectorXd getRandomPoint() {
    Eigen::VectorXd out(2);
    out(0) = long(R::runif(0, 1) * cols);
    out(1) = 0.;
    return out;
  }

  // First coordinate is the desired row and second is irrelevant.
  Eigen::MatrixXd getVariables(const Eigen::MatrixXd& coordinates,
                               std::vector<int> columns) {
    Eigen::MatrixXd out(coordinates.rows(), columns.size());
    for (int i = 0; i < coordinates.rows(); i++)
      for (int j = 0; j < columns.size(); j++)
        out(i, j) = data[j][i];
    return out;
  }
};

#endif
