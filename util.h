#pragma once

#include <Eigen/Core>
#include <gsl/gsl_randist.h>
#include <vector>

inline Eigen::VectorXd GaussianVector(gsl_rng* rng, int d) {
  Eigen::VectorXd result(d);
  for (int i = 0; i < d; i++) {
    result[i] = gsl_ran_gaussian_ziggurat(rng, 1.0);
  }
  return result;
}

void subsample_path(const std::vector<double>& x,
		    const std::vector<double>& v,
		    const std::vector<double>& t,
		    int num_samples,
		    std::vector<double>* x_samples_out);
