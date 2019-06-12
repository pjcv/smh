#include "fmh.h"

#include <boost/timer/timer.hpp>
#include <cmath>
#include <Eigen/Core>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "alias.h"

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::make_unique;
using std::unique_ptr;
using std::vector;

double coefficient(int taylor_order,
                   const VectorXd& mode,
                   const VectorXd& theta1,
                   const VectorXd& theta2) {
  if (taylor_order == 1) {
    return pow((theta1 - mode).array().abs().sum(), 2.0) / 2.0
      + pow((theta2 - mode).array().abs().sum(), 2.0) / 2.0;
  } else if (taylor_order == 2) {
    return pow((theta1 - mode).array().abs().sum(), 3.0) / 6.0
      + pow((theta2 - mode).array().abs().sum(), 3.0) / 6.0;
  }

  assert(false);
}

void fmh(gsl_rng* rng,
         const Target* target,
         const VectorXd& mode,
         int taylor_order,
         Proposal* proposal,
         const VectorXd& theta0,
         StoppingRule* stopping_rule,
         Observer* observer) {
  vector<double> bounds = target->Bounds(taylor_order);
  double sum_bounds = std::accumulate(bounds.begin(), bounds.end(), 0.0);
  unique_ptr<Alias> alias = make_unique<Alias>(rng, bounds);

  VectorXd gradient_at_mode = target->Gradient(mode);
  MatrixXd hessian_at_mode = target->Hessian(mode);

  vector<VectorXd> gradients_at_mode;
  vector<MatrixXd> hessians_at_mode;
  for (int i = 0; i < target->Count(); i++) {
    gradients_at_mode.push_back(target->Gradient(i, mode));
    hessians_at_mode.push_back(target->Hessian(i, mode));
  }

  VectorXd current_theta = theta0;
  double previous_iteration_fellback = false;
  double previous_fallback_potential = NAN;

  int num_likelihood_evaluations = 0;
  observer->OnStart();
  stopping_rule->Start();
  boost::timer::auto_cpu_timer t("MCMC: %u user, %s system, %w wall\n");
  for (int it = 0; stopping_rule->ShouldContinue(it); it++) {
    if (it % 1000 == 0) std::cout << "Iteration " << it << "\r";

    VectorXd proposed_theta;
    double proposal_potential_difference;
    proposal->Sample(current_theta, &proposed_theta, &proposal_potential_difference);

    double coeff = coefficient(taylor_order, mode, current_theta, proposed_theta);
    double poisson_rate = sum_bounds * coeff;

    if (poisson_rate > target->Count()) {
      double current_fallback_potential;
      if (previous_iteration_fellback) {
        current_fallback_potential = previous_fallback_potential;
      } else {
        current_fallback_potential = target->Potential(current_theta);
      }

      double proposed_fallback_potential = target->Potential(proposed_theta);

      if (gsl_rng_uniform(rng) < exp(current_fallback_potential - proposed_fallback_potential
                                     - proposal_potential_difference)) {
        current_theta = proposed_theta;
        previous_fallback_potential = proposed_fallback_potential;
      } else {
        previous_fallback_potential = current_fallback_potential;
      }
      previous_iteration_fellback = true;
    } else {
      double potential_difference = gradient_at_mode.dot(proposed_theta - current_theta);
      if (taylor_order == 2) {
        potential_difference +=
          0.5 * (proposed_theta - mode).dot(hessian_at_mode * (proposed_theta - mode))
          - 0.5 * (current_theta - mode).dot(hessian_at_mode * (current_theta - mode));
      }

      double acceptance_probability = exp(-potential_difference - proposal_potential_difference);
      if (gsl_rng_uniform(rng) < acceptance_probability) {
        bool reject = false;
	int N = gsl_ran_poisson(rng, poisson_rate);
	for (int i = 0; i < N; i++) {
	  int idx = alias->Sample();
	  double nlap_i = target->Potential(idx, proposed_theta)
	    - target->Potential(idx, current_theta)
	    + gradients_at_mode[idx].dot(current_theta - proposed_theta);
	  if (taylor_order == 2) {
	    nlap_i += 0.5 * (current_theta - mode).dot(hessians_at_mode[idx] * (current_theta - mode))
	      - 0.5 * (proposed_theta - mode).dot(hessians_at_mode[idx] * (proposed_theta - mode));
	  }

	  if (nlap_i > 0.0) {
	    double nlap_bound = coeff * bounds[idx];
	    assert(nlap_i <= nlap_bound);
	    if (gsl_rng_uniform(rng) < nlap_i / nlap_bound) {
	      reject = true;
	      break;
	    }
	  }
	}
        if (!reject) {
          current_theta = proposed_theta;
          previous_iteration_fellback = false;
        }
      }
    }

    observer->OnSample(current_theta);
  }

  observer->OnComplete();
}
