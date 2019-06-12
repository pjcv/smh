#include "flymc.h"

#include <boost/timer/timer.hpp>
#include <Eigen/Core>
#include <unordered_set>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::make_pair;
using std::pair;
using std::unordered_set;
using std::vector;


bool contains(const unordered_set<int>& s, int e) {
  return s.find(e) != s.end();
}

template <typename T>
void remove(vector<T>& v, int idx) {
  v[idx] = v[v.size() - 1];
  v.pop_back();
}

void flymc(gsl_rng* rng,
           const Target* target,
           const VectorXd& mode,
           Proposal* proposal,
           double qdb,
           const VectorXd& theta0,
           StoppingRule* stopping_rule,
           Observer* observer) {
  int N = target->Count();
  int d = target->Dimension();

  VectorXd theta_current = theta0;

  // Data indices separated by brightness.
  vector<pair<int, double>> bright;
  vector<int> dark;
  for (int i = 0; i < N; i++) {
    dark.push_back(i);
  }

  VectorXd mean;
  MatrixXd precision;
  double constant;

  target->LowerBoundLogDensityParams(mode, &mean, &precision, &constant);

  vector<VectorXd> means;
  vector<MatrixXd> precisions;
  vector<double> constants;
  for (int i = 0; i < N; i++) {
    VectorXd mean_i;
    MatrixXd precision_i;
    double constant_i;
    target->LowerBoundLogDensityParams(i, mode, &mean_i, &precision_i, &constant_i);
    means.push_back(mean_i);
    precisions.push_back(precision_i);
    constants.push_back(constant_i);
  }

  observer->OnStart();
  stopping_rule->Start();
  boost::timer::auto_cpu_timer t("MCMC: %u user, %s system, %w wall\n");
  for (int it = 0; stopping_rule->ShouldContinue(it); it++) {
    if (it % 1000 == 0) std::cout << "iteration " << it << "\r";
    // Implicit sampling.

    // Consider darkening each of the bright data points.
    vector<int> newly_dark;
    for (int i = 0; i < bright.size(); ) {
      if (gsl_rng_uniform(rng) < qdb / bright[i].second) {
        newly_dark.push_back(bright[i].first);
        remove(bright, i);
      } else {
        i++;
      }
    }

    // Now consider brightening each of the previously dark data points.
    int k = gsl_ran_geometric(rng, qdb) - 1;
    while (k < dark.size()) {
      int n = dark[k];

      double log_B_n = theta_current.dot(precisions[n] * theta_current)
        + means[n].dot(theta_current)
        + constants[n];
      double B_n = exp(log_B_n);
      double L_n = exp(-target->Potential(n, theta_current));
      double L_n_tilde = (L_n - B_n) / B_n;
      if (gsl_rng_uniform(rng) < L_n_tilde / qdb) {
        bright.push_back(make_pair(n, L_n_tilde));
        remove(dark, k);
      } else {
        k += 1;
      }
      k += gsl_ran_geometric(rng, qdb) - 1;
    }
    dark.insert(dark.end(), newly_dark.begin(), newly_dark.end());

    VectorXd theta_proposal;
    double proposal_potential_difference;
    proposal->Sample(theta_current, &theta_proposal, &proposal_potential_difference);

    double log_density_current = theta_current.dot(precision * theta_current)
      + mean.dot(theta_current)
      + constant;

    double log_density_proposal =  theta_proposal.dot(precision * theta_proposal)
      + mean.dot(theta_proposal)
      + constant;

    vector<double> proposal_L_n_tildes;
    for (int i = 0; i < bright.size(); i++) {
      log_density_current += log(bright[i].second);
      double B_n = exp(theta_proposal.dot(precisions[bright[i].first] * theta_proposal)
                       + means[bright[i].first].dot(theta_proposal)
                       + constants[bright[i].first]);
      double L_n = exp(-target->Potential(bright[i].first, theta_proposal));
      double proposal_L_n_tilde = (L_n - B_n) / B_n;
      log_density_proposal += log(proposal_L_n_tilde);
      proposal_L_n_tildes.push_back(proposal_L_n_tilde);
    }

    if (gsl_rng_uniform(rng) < exp(log_density_proposal - log_density_current
                                   - proposal_potential_difference)) {
      theta_current = theta_proposal;
      for (int i = 0; i < bright.size(); i++) {
        bright[i].second = proposal_L_n_tildes[i];
      }
    }

    observer->OnSample(theta_current);
  }

  observer->OnComplete();
}
