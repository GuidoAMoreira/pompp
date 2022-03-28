#ifndef __POMPP_COVARIANCE_FUNCTION_H__
#define __POMPP_COVARIANCE_FUNCTION_H__

#include <RcppEigen.h>
#include "GaussianProcess.hpp"

class CovarianceFunction {
  const double maxDist;
  double logDensity;
public:
  CovarianceFunction(double mD, double s2) : maxDist(mD), sigma2(s2) {}

  // getters
  virtual double getPar(int index) = 0;
  double getLogDensity() {return logDensity;}
  virtual int getParSize() = 0;
  double getSigma2() {return sigma2;}
  // setters
  virtual void setPar(double newValue, int index) = 0;
  void setLogDensity(double newValue) {logDensity = newValue;}

  virtual double operator()(double dist, Eigen::VectorXd pars) = 0;
  virtual double operator()(double dist) = 0;
  virtual double calcRange(double maxCor, Eigen::VectorXd pars) = 0;
  virtual double calcRange(double maxCor) = 0;
  double calcRange(Eigen::VectorXd pars) {double fromCov = calcRange(0.05, pars); return fromCov > maxDist ? maxDist : fromCov;}
protected:
  double sigma2;
};

// Usable classes
class PowerExponentialCovariance : public CovarianceFunction {
  double phi;
  const double alpha;
public:
  PowerExponentialCovariance(double mD, double p, double a, double s2) :
    CovarianceFunction(mD, s2), phi(p), alpha(a) {
    if (a <= 0 || a > 2) Rf_error("Invalid alpha value for the exponential covariance function. Needs to be in (0, 2].");
  }

  // getters
  double getPar(int index = 0) {return phi;}
  int getParSize() {return 1;}
  // setters
  void setPar(double newValue, int index = 0) {phi = newValue;}

  double operator()(double dist, Eigen::VectorXd pars) {return sigma2 * exp(-pow(dist, alpha) / pars(0));}
  double operator()(double dist) {Eigen::VectorXd p(1); p(0) = phi; return calcCov(dist, p);}
  double calcRange(double maxCor, Eigen::VectorXd pars) {return pow(-pars(0) * log(maxCor), 1 / alpha);}
  double calcRange(double maxCor) {Eigen::VectorXd p(1); p(0) = phi; return calcRange(maxCor, p);}
};

#endif
