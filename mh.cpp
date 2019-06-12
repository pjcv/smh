#include "mh.h"

#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include <iostream>

using Eigen::VectorXd;

void mh(gsl_rng* rng,
        const Target* target,
        Proposal* proposal,
        const VectorXd& theta0,
        StoppingRule* stopping_rule,
        Observer* observer) {
  VectorXd current_theta = theta0;
  double current_potential = target->Potential(current_theta);

  boost::timer::auto_cpu_timer t("MCMC: %u user, %s system, %w wall\n");
  observer->OnStart();
  stopping_rule->Start();
  for (int it = 0; stopping_rule->ShouldContinue(it); it++) {
    if (it % 1000 == 0)  std::cout << "iteration: " << it << "\r";
    VectorXd proposed_theta;
    double proposal_potential_difference;
    proposal->Sample(current_theta, &proposed_theta, &proposal_potential_difference);

    double proposed_potential = target->Potential(proposed_theta);

    if (gsl_rng_uniform(rng) < exp(current_potential - proposed_potential
                                   - proposal_potential_difference)) {
      current_theta = proposed_theta;
      current_potential = proposed_potential;
    }

    observer->OnSample(current_theta);
  }
  observer->OnComplete();
}
