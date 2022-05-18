#include <RcppEigen.h>
#include "include/PresenceOnly.hpp"
#include "include/BackgroundVariables.hpp"
#include "include/BinaryRegression.hpp"
#include "include/RegressionPrior.hpp"
#include <progress.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
List cppPOMPP(Eigen::VectorXd beta, Eigen::VectorXd delta,
                   double lambda, Rcpp::String b_updater,
                   Rcpp::String d_updater, Rcpp::String l_updater,
                   Rcpp::List parB, Rcpp::List parD,
                   double lambdaA, double lambdaB,
                   Rcpp::String covsClass, SEXP covariates,
                   double areaD, Rcpp::String xClass,
                   double marksMuMu, double marksMuS2,
                   double marksNuggetA, double marksNuggetB,
                   double marksShapeA, double marksShapeB,
                   Eigen::MatrixXd xValues, Eigen::VectorXd xMarks,
                   Eigen::MatrixXd xPositions,
                   Eigen::VectorXi intensityCovs,
                   Eigen::VectorXi observabilityCovs,
                   Eigen::VectorXi xIntensityCovs,
                   Eigen::VectorXi xObservabilityCovs,
                   int longCol, int latCol,
                   int burnin, int thin, int iter, int threads, bool verbose) {
  int i, j;

  // Auxiliary
  Eigen::MatrixXd xInt(xValues.rows(), xIntensityCovs.size());
  for (i = 0; i < xValues.rows(); i++)
    for (j = 0; j < xIntensityCovs.size(); j++)
      xInt(i, j) = xValues(i, xIntensityCovs(j));
  Eigen::MatrixXd xObs(xValues.rows(), xObservabilityCovs.size());
  for (i = 0; i < xValues.rows(); i++)
    for (j = 0; j < xObservabilityCovs.size(); j++)
      xObs(i, j) = xValues(i, xObservabilityCovs(j));

  // Output storage
  int outSize = iter / thin;
  Eigen::MatrixXd outBetas(outSize, beta.size());
  Eigen::MatrixXd outDeltas(outSize, delta.size());
  Eigen::VectorXd outLambdas(outSize);
  Eigen::VectorXd outShapes(outSize);
  Eigen::VectorXd outMus(outSize);
  Eigen::VectorXd outNuggets(outSize);
  Eigen::VectorXd outLogPost(outSize);
  Eigen::VectorXd out_nU(outSize);
  Eigen::VectorXd out_nXp(outSize);
  Eigen::VectorXd outMarksPrimeMean(outSize);
  Eigen::VectorXd outMarksPrimeVariance(outSize);
  Eigen::VectorXd outAllMarksMean(outSize);
  Eigen::VectorXd outAllMarksVariance(outSize);

  // Get prior parameters
  Eigen::VectorXd muB = parB["mean"];
  Eigen::MatrixXd SigmaB = parB["covariance"];
  Eigen::VectorXd muD = parB["mean"];
  Eigen::MatrixXd SigmaD = parB["covariance"];

  PresenceOnly mc(
    xPositions, xInt, xObs,
    new MatrixVariables(
      std::vector<int>(&intensityCovs[0], intensityCovs.data() + intensityCovs.size()),
      std::vector<int>(&observabilityCovs[0], observabilityCovs.data() + observabilityCovs.size()),
      covariates, longCol, latCol
    ), xMarks,
    new LogisticRegression(beta, new NormalPrior(muB, SigmaB)),
    new LogisticRegression(delta, new NormalPrior(muD, SigmaD)),
    lambda, lambdaA, lambdaB,
    areaD, marksMuMu, marksMuS2, marksNuggetA, marksNuggetB,
    marksShapeA, marksShapeB
  );

  // Burning in
  if (burnin) {
    if (verbose) Rcpp::Rcout << "Warming up the Markov Chain.\n";
    Progress progr_Burnin(burnin, true);
    for (i = 0; i < burnin; i++)
    {
      progr_Burnin.increment();
      mc.update();
    }
    if (verbose) Rcpp::Rcout << "Warm up complete. ";
  }

  if (verbose) Rcpp::Rcout << "Sampling MCMC.\n";
  double xMarksPrimeSquaredNorm, xMarksSquaredNorm = xMarks.squaredNorm();
  int fullSize;
  Progress progr_Main(outSize, true);
  for (i = 0; i < outSize; i++)
  {
    R_CheckUserInterrupt();
    mc.update(thin);
    outBetas.row(i) = mc.getBeta();
    outDeltas.row(i) = mc.getDelta();
    outLambdas[i] = mc.getLambdaStar();
    outShapes[i] = mc.getMarksShape();
    outMus[i] = mc.getMarksMu();
    outNuggets[i] = mc.getMarksNugget();
    out_nU[i] = mc.getUsize();
    out_nXp[i] = mc.getXpsize();
    fullSize = xMarks.size() + mc.getXpsize();
    outMarksPrimeMean[i] = mc.getMarksPrime().mean();
    xMarksPrimeSquaredNorm = mc.getMarksPrime().squaredNorm();
    outMarksPrimeVariance[i] = xMarksPrimeSquaredNorm / mc.getXpsize() -
      outMarksPrimeMean[i] * outMarksPrimeMean[i];
    outAllMarksMean[i] = (outMarksPrimeMean[i] * mc.getXpsize() + xMarks.sum()) /
      fullSize;
    outAllMarksVariance[i] = (xMarksPrimeSquaredNorm + xMarksSquaredNorm) / fullSize -
      outAllMarksMean[i] * outAllMarksMean[i];
    outLogPost[i] = mc.getLogPosterior();
    progr_Main.increment();
  }
  if (verbose) Rcpp::Rcout << "MCMC sampling complete.\n";

//  delete mc;

  return Rcpp::List::create(Rcpp::Named("beta") = outBetas,
                            Rcpp::Named("delta") = outDeltas,
                            Rcpp::Named("lambda") = outLambdas,
                            Rcpp::Named("shape") = outShapes,
                            Rcpp::Named("mu") = outMus,
                            Rcpp::Named("nugget") = outNuggets,
                            Rcpp::Named("nU") = out_nU,
                            Rcpp::Named("nXp") = out_nXp,
                            Rcpp::Named("marksPrimeMean") = outMarksPrimeMean,
                            Rcpp::Named("marksPrimeVariance") = outMarksPrimeVariance,
                            Rcpp::Named("allMarksMean") = outAllMarksMean,
                            Rcpp::Named("allMarksVariance") = outAllMarksVariance,
                            Rcpp::Named("logPost") = outLogPost);
}
