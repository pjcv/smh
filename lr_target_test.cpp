#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE lr target test

#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <Eigen/Core>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <vector>

#include "lr_target.h"
#include "target_test.h"
#include "util.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

LrTarget create_target(int n, int d) {
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  VectorXd labels(n);
  MatrixXd covariates(n, d);
  for (int i = 0; i < n; i++) {
    labels[i] = gsl_rng_uniform(rng) < 0.5 ? -1.0 : 1.0;
    for (int j = 0; j < d; j++) {
      covariates(i, j) = gsl_ran_gaussian_ziggurat(rng, 1.0);
    }
  }
  
  return LrTarget(labels, covariates);
}

BOOST_AUTO_TEST_CASE(large_potential_test) {
  int d = 2;
  LrTarget target = create_target(10000, d);

  VectorXd theta1(d);
  for (int i = 0; i < d; i++) {
    theta1[i] = 10000;
  }

  double U = target.Potential(theta1);

  BOOST_CHECK(std::isfinite(U));
}

BOOST_AUTO_TEST_CASE(potential_benchmarking) {
  int n = 1000000;

  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  for (int d : {10, 100}) {
    try {
      LrTarget target = create_target(n, d);

      VectorXd theta = GaussianVector(rng, d);

      boost::timer::auto_cpu_timer timer("%u user, %s system, %w wall\n");
      double U = target.Potential(theta);
      std::cout << "d: " << d << std::endl;
    } catch (const std::bad_alloc&) {}
  }
}

BOOST_AUTO_TEST_CASE(gradient_test) {
  LrTarget t = create_target(100, 2);
  test_gradient(t);
}

BOOST_AUTO_TEST_CASE(gradient_coordinate_test) {
  LrTarget t = create_target(100, 2);
  test_gradient_coordinate(t);
}

BOOST_AUTO_TEST_CASE(hessian_test) {
  LrTarget t = create_target(4, 2);
  test_hessian(t);
}
