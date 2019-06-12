#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rlr target test

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <gsl/gsl_randist.h>
#include <vector>

#include "rlr_target.h"
#include "target_test.h"
#include "util.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

RlrTarget create_target() {
  int n = 100;
  int d = 2;

  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  VectorXd labels(n);
  MatrixXd covariates(n, d);
  for (int i = 0; i < n; i++) {
    labels[i] = gsl_ran_gaussian(rng, 1.0);
    for (int j = 0; j < d; j++) {
      covariates(i, j) = gsl_ran_gaussian(rng, 1.0);
    }
  }
  RlrTarget target(labels, covariates, 4.0);
  return target;
}

BOOST_AUTO_TEST_CASE(gradient_test) {
  RlrTarget t = create_target();
  test_gradient(t);
}

BOOST_AUTO_TEST_CASE(gradient_coordinate_test) {
  RlrTarget t = create_target();
  test_gradient_coordinate(t);
}

BOOST_AUTO_TEST_CASE(hessian_test) {
  RlrTarget t = create_target();
  test_hessian(t);
}
