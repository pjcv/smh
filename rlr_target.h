#pragma once

#include <Eigen/Core>

#include "target.h"


class RlrTarget : public Target {
 public:
  RlrTarget(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, double nu)
    : num_potential_evaluations_(0),
      y_(y),
      X_(X),
      nu_(nu)
  {}

  virtual int Count() const {
    return X_.rows();
  }

  virtual int Dimension() const {
    return X_.cols();
  }

  virtual int64_t NumPotentialEvaluations() const {
    return num_potential_evaluations_;
  }

  virtual double y(int n) const {
    return y_[n];
  }

  virtual Eigen::VectorXd X(int n) const {
    return X_.row(n);
  }
  
  virtual double Potential(const Eigen::VectorXd& theta) const {
    double rv = 0.0;
    for (int i = 0; i < X_.rows(); i++) {
      rv += Potential(i, theta);
    }
    return rv;
  }

  virtual double Potential(int n, const Eigen::VectorXd& theta) const {
    num_potential_evaluations_ += 1;
    double phi = y_[n] - theta.dot(X_.row(n));
    return 0.5 * (nu_ + 1.0) * log(1.0 + phi * phi / nu_);
  }

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) const {
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(theta.size());
    for (int i = 0; i < X_.rows(); i++) {
      grad += Gradient(i, theta);
    }
    return grad;
  }

  virtual Eigen::VectorXd Gradient(int n, const Eigen::VectorXd& theta) const {
    double phi = y_[n] - theta.dot(X_.row(n));
    return -(nu_ + 1.0) * X_.row(n) * phi / (nu_ + phi * phi);    
  }

  
  virtual double GradientCoordinate(int n, int d, const Eigen::VectorXd& theta) const {
    double phi = y_[n] - theta.dot(X_.row(n));
    return -(nu_ + 1.0) * X_(n, d) * phi / (nu_ + phi * phi);    
  }


  Eigen::MatrixXd Hessian(const Eigen::VectorXd& theta) const {
    int d = theta.size();
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(d, d);
    for (int i = 0; i < X_.rows(); i++) {
      hessian += Hessian(i, theta);
    }
    return hessian;
  }

  Eigen::MatrixXd Hessian(int n, const Eigen::VectorXd& theta) const {
    Eigen::VectorXd Xn = X_.row(n);
    double phi = y_[n] - theta.dot(Xn);
    return (nu_ + 1) * (nu_ - phi * phi) / pow(nu_ + phi * phi, 2.0) * Xn * Xn.transpose();
  }

  virtual std::vector<double> Bounds(int taylor_order) const {
    std::vector<double> rv;
    for (int i = 0; i < X_.rows(); i++) {
      double max_abs_coef = X_.row(i).array().abs().maxCoeff();
      switch (taylor_order) {
      case 1:
        rv.push_back(pow(max_abs_coef, 2.0) * (nu_ + 1.0) / nu_);
        break;
      case 2:
        rv.push_back(pow(max_abs_coef, 3.0) * 2.0 * (nu_ + 1.0) * (3.0 + 2.0 * sqrt(2.0)) / (8.0 * pow(nu_, 1.5)));
        break;
      default:
        assert(false);
        break;
      }
    }
    return rv;
  }

  virtual void LowerBoundLogDensityParams(const Eigen::VectorXd& mode,
                                          Eigen::VectorXd* mean,
                                          Eigen::MatrixXd* precision,
                                          double* constant) const {
    int d = Dimension();
    *precision = Eigen::MatrixXd::Zero(d, d);
    *mean = Eigen::VectorXd::Zero(d);
    *constant = 0.0;
    for (int i = 0; i < X_.rows(); i++) {
      Eigen::VectorXd Xi = X_.row(i);
      double phi = y_[i] - Xi.dot(mode);
      *precision += a(phi) * Xi * Xi.transpose();
      *mean += -2.0 * y_[i] * a(phi) * Xi;
      *constant += c(phi) + y_[i] * a(phi) * y_[i];
    }
  }

  virtual void LowerBoundLogDensityParams(int i,
                                          const Eigen::VectorXd& mode,
                                          Eigen::VectorXd* mean,
                                          Eigen::MatrixXd* precision,
                                          double* constant) const {
    Eigen::VectorXd Xi = X_.row(i);
    double phi = y_[i] - Xi.dot(mode);
    *precision = a(phi) * Xi * Xi.transpose();
    *mean = -2.0 * y_[i] * a(phi) * Xi;
    *constant = c(phi) + y_[i] * a(phi) * y_[i];
  }

  
 private: 
  Eigen::VectorXd y_;
  CovariateT X_;
  double nu_;

  mutable int64_t num_potential_evaluations_;

  double a(double phi) const {
    return -0.5 * (nu_ + 1.0) / (nu_ + phi * phi);
  }
  
  double b(double phi) const {
    return -2.0 * a(phi);
  }
  
  double c(double phi) const {
    return -a(phi) * phi * phi - 0.5 * (nu_ + 1.0) * log(1.0 + phi * phi / nu_);
  }
};
