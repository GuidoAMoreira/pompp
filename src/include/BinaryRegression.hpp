#ifndef __pompp_BINARY_REGRESSION_HPP__
#define __pompp_BINARY_REGRESSION_HPP__

#include <RcppEigen.h>
#include "RegressionPrior.hpp"

class BinaryRegression {
protected:
  Eigen::VectorXd betas, normalMean; // normalMean is used in the GP part of the program.
  RegressionPrior* prior;
  const unsigned int n;

public:
  virtual double sample(const Eigen::MatrixXd& onesCovariates,
                        const Eigen::MatrixXd& zerosCovariates) = 0;


  // Link function. Returns in the **LOG** scale. complementaryProb = true
  // calculates the link for 1 - p
  // For the stored beta vector
  Eigen::VectorXd link(const Eigen::MatrixXd covariates,
                       bool complementaryProb = false) {
    return link(covariates, betas, complementaryProb);
  }

  // For any beta vector
  virtual Eigen::VectorXd link(const Eigen::MatrixXd covariates,
                               const Eigen::VectorXd beta,
                               bool complementaryProb = false) = 0;

  // Constructor
  BinaryRegression(Eigen::VectorXd initialize) : betas(initialize),
    n(initialize.size()) {}

  // Prior setter
  void setPrior(RegressionPrior* p) {prior = p;}

  // Some getters
  Eigen::VectorXd getBeta() {return betas;}
  int getSize() {return n;}
  Eigen::VectorXd getNormalMean() {return normalMean;}
  virtual Eigen::VectorXd getExtra() = 0; // For data augmentation variables
  // Some setters
  void setNormalMean(Eigen::VectorXd newValue) {normalMean = newValue;}
  void setNormalMeanIndex(double newValue, long index) {normalMean(index) = newValue;}
};

class LogisticRegression : public BinaryRegression {
  // Data augmentation
  std::vector<double> pg;

  // Necessary getter
  std::vector<double> getPolyaGamma() {return pg;}
public:
  Eigen::VectorXd getExtra() {return normalMean;}

  LogisticRegression(Eigen::VectorXd initialize) : BinaryRegression(initialize) {}

  double sample(const Eigen::MatrixXd& onesCovariates,
                const Eigen::MatrixXd& zerosCovariates);
  Eigen::VectorXd link(const Eigen::MatrixXd& covariates,
                       const Eigen::VectorXd& beta,
                       bool complementaryProb);
};

#endif
