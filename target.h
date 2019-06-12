#pragma once

#include <Eigen/Core>
#include <json/json.h>


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CovariateT;

class Target {
 public:
  virtual ~Target() {};
  
  virtual int Count() const = 0;
  virtual int Dimension() const = 0;

  virtual int64_t NumPotentialEvaluations() const = 0;
  
  virtual double Potential(const Eigen::VectorXd& theta) const = 0;
  virtual double Potential(int n, const Eigen::VectorXd& theta) const = 0;

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) const = 0;
  virtual Eigen::VectorXd Gradient(int n, const Eigen::VectorXd& theta) const = 0;
  virtual double GradientCoordinate(int n, int d, const Eigen::VectorXd& theta) const = 0;

  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& theta) const = 0;
  virtual Eigen::MatrixXd Hessian(int n, const Eigen::VectorXd& theta) const = 0;

  // For Factorized MH.
  virtual std::vector<double> Bounds(int taylor_order) const = 0;


  // For Firefly MCMC.
  virtual void LowerBoundLogDensityParams(const Eigen::VectorXd& mode,
                                          Eigen::VectorXd* mean,
                                          Eigen::MatrixXd* precision,
                                          double* constant) const = 0;

  virtual void LowerBoundLogDensityParams(int i,
                                          const Eigen::VectorXd& mode,
                                          Eigen::VectorXd* mean,
                                          Eigen::MatrixXd* precision,
                                          double* constant) const = 0;
};
