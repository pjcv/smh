#pragma once

#include "target.h"


class LrTarget : public Target {
 public:
  LrTarget(const Eigen::VectorXd& y, const Eigen::MatrixXd& X)
    : num_potential_evaluations_(0),
      y_(y),
      X_(X)
  {}

  Eigen::VectorXd X(int i) const {
    return X_.row(i);
  }

  virtual int Count() const {
    return X_.rows();
  }

  virtual int Dimension() const {
    return X_.cols();
  }

  virtual int64_t NumPotentialEvaluations() const {
    return num_potential_evaluations_;
  }
  
  virtual double Potential(const Eigen::VectorXd& theta) const {
    num_potential_evaluations_ += X_.rows();

    Eigen::VectorXd w = y_.cwiseProduct(X_ * theta);
    double nll = 0.0;
    for (int n = 0; n < X_.rows(); n++) {
      if (w[n] < 0) {
        nll += log(exp(w[n]) + 1.0) - w[n];
      } else {
        nll += log(1.0 + exp(-w[n]));
      }
    }
    return nll;    
  }

  virtual double Potential(int n, const Eigen::VectorXd& theta) const {
    num_potential_evaluations_ += 1;

    double w = y_(n) * X_.row(n).dot(theta);
    if (w < 0) {
      return log(exp(w) + 1.0) - w;
    } else {
      return log(1.0 + exp(-w));
    }
  }

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& theta) const {
    Eigen::VectorXd w = y_.cwiseProduct(X_ * theta);
    Eigen::VectorXd rv = Eigen::VectorXd::Zero(theta.size());
    for (int i = 0; i < y_.size(); i++) {
      rv += y_(i) * X_.row(i) / (exp(w[i]) + 1.0);
    }
    return -rv;
  }

  virtual Eigen::VectorXd Gradient(int n, const Eigen::VectorXd& theta) const {
    double w = y_(n) * X_.row(n).dot(theta);
    return -X_.row(n) * y_(n) / (exp(w) + 1.0);
  }

  virtual double GradientCoordinate(int n, int d, const Eigen::VectorXd& theta) const {
    double w = y_(n) * X_.row(n).dot(theta);
    return -X_(n, d) * y_(n) / (exp(w) + 1.0);
  }

  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& theta) const {
    Eigen::VectorXd w = X_ * theta;
    Eigen::ArrayXd expw = exp(w.array());
    int d = X_.cols();
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(d, d);
    for (int j = 0; j < d; j++) {
      for (int k = 0; k < d; k++) {
        for (int i = 0; i < X_.rows(); i++) {
          rv(j, k) -= X_(i, j) * X_(i, k) * expw[i] / (expw[i] + 1.0) / (expw[i] + 1.0);
        }
      }
    }
    return -rv;
  }
  
  virtual Eigen::MatrixXd Hessian(int n, const Eigen::VectorXd& theta) const {
    double w = X_.row(n).dot(theta);
    double expw = exp(w);
    int d = X_.cols();
    Eigen::MatrixXd rv = Eigen::MatrixXd::Zero(d, d);
    for (int j = 0; j < d; j++) {
      for (int k = 0; k < d; k++) {
        rv(j, k) -= X_(n, j) * X_(n, k) * expw / (expw + 1.0) / (expw + 1.0);
      }
    }
    return -rv;
  }

  virtual std::vector<double> Bounds(int taylor_order) const {
    std::vector<double> rv;
    for (int i = 0; i < X_.rows(); i++) {
      double max_abs_coef = X_.row(i).array().abs().maxCoeff();
      switch (taylor_order) {
      case 1:
        rv.push_back(pow(max_abs_coef, 2.0) / 4.0);
        break;
      case 2:
        rv.push_back(pow(max_abs_coef, 3.0) / 6.0 / sqrt(3.0));
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
    Eigen::MatrixXd S_hat = Eigen::MatrixXd::Zero(d, d);
    Eigen::VectorXd mu_hat = Eigen::VectorXd::Zero(d);
    double c_hat = 0.0;
    for (int i = 0; i < X_.rows(); i++) {
      Eigen::VectorXd Xi = X_.row(i);
      double xi = -y_(i) * Xi.dot(mode);
      S_hat += a(xi) * Xi * Xi.transpose();
      mu_hat += b(xi) * y_(i) * Xi;
      c_hat += c(xi);
    }
    *mean = mu_hat;
    *precision = S_hat;
    *constant = c_hat;
  }

  virtual void LowerBoundLogDensityParams(int i,
                                          const Eigen::VectorXd& mode,
                                          Eigen::VectorXd* mean,
                                          Eigen::MatrixXd* precision,
                                          double* constant) const {
    Eigen::VectorXd Xi = X_.row(i);
    double xi = -y_(i) * Xi.dot(mode);
    *precision = a(xi) * Xi * Xi.transpose();
    *mean = b(xi) * y_(i) * Xi;
    *constant = c(xi);    
  }
  
 private:

  double a(double xi) const {
    return -(exp(xi) - 1) / (exp(xi) + 1) / 4.0 / xi;
  }
  
  double b(double xi) const {
    return 0.5;
  }
  
  double c(double xi) const {
    return -a(xi) * xi * xi + 0.5 * xi - log(exp(xi) + 1.0);
  }
  
  Eigen::VectorXd y_;
  CovariateT X_;
  
  mutable int64_t num_potential_evaluations_;
};
