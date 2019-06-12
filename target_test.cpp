#include "target_test.h"

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <gsl/gsl_randist.h>

#include "util.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

void test_gradient(const Target& target) {
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  int d = target.Dimension();
  VectorXd theta = GaussianVector(rng, d);

  VectorXd actual_gradient = target.Gradient(theta);

  double U = target.Potential(theta);
  double h = 0.0001;
  for (int i = 0; i < d; i++) {
    VectorXd pert_theta = theta;
    pert_theta(i) += h;
    double pert_U = target.Potential(pert_theta);
    double estimated_gradient = (pert_U - U) / h;
    BOOST_CHECK_CLOSE(actual_gradient(i), estimated_gradient, 0.1);
  }
}

void test_gradient_coordinate(const Target& target) {
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  int n = target.Count();
  int d = target.Dimension();
  VectorXd theta = GaussianVector(rng, d);

  for (int i = 0; i < n; i++) {
    VectorXd vector_gradient = target.Gradient(i, theta);
    for (int j = 0; j < d; j++) {
      double coordinate_gradient = target.GradientCoordinate(i, j, theta);
      BOOST_CHECK_CLOSE(coordinate_gradient, vector_gradient(j), 1e-6);
    }
  }
}

void test_hessian(const Target& target) {
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);
  
  int d = target.Dimension();
  VectorXd theta = GaussianVector(rng, d);

  MatrixXd actual_hessian = target.Hessian(theta);

  double U = target.Potential(theta);
  double h = 0.0001;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      VectorXd theta1 = theta;
      theta1[i] += h;
      theta1[j] += h;

      VectorXd theta2 = theta;
      theta2[i] += h;

      VectorXd theta3 = theta;
      theta3[j] += h;

      double U_p = target.Potential(theta1)
        - target.Potential(theta2)
        - target.Potential(theta3)
        + U;
      BOOST_CHECK_CLOSE(actual_hessian(i, j), U_p / h / h, 0.1);
    }
  }
}
