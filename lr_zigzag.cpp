#include "lr_zigzag.h"

#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

double quadratic_solution(double a, double b, double c) {
  double det = b * b - 4 * a * c;
  assert(det >= 0.0);
  return (-b + sqrt(det)) / 2.0 / a;
}

void lr_zigzag(gsl_rng* rng,
               const LrTarget* target,
               const Eigen::VectorXd& mode,
               const VectorXd& theta0,
               StoppingRule* stopping_rule,
               ContinuousTimeObserver* observer) {
  int d = target->Dimension();
  int N = target->Count();

  VectorXd global_gradient_at_mode = target->Gradient(mode);
  vector<VectorXd> target_gradients_at_mode;
  for (int i = 0; i < N; i++) {
    target_gradients_at_mode.push_back(target->Gradient(i, mode));
  }

  // Compute C_i as described in the supplementary material.
  vector<double> C;
  for (int i = 0; i < d; i++) {
    double max = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < N; j++) {
      VectorXd Xj = target->X(j);
      max = std::max(max, fabs(Xj[i]) * Xj.norm());
    }
    C.push_back(N * 0.25 * max);
  }

  VectorXd theta = theta0;
  VectorXd v(d);
  for (int i = 0; i < d; i++) {
    if (gsl_rng_uniform(rng) < 0.5) {
      v[i] = -1.0;
    } else {
      v[i] = 1.0;
    }
  }

  // The b vector is constant as the norm of v does not change.
  vector<double> b;
  for (int i = 0; i < d; i++) {
    b.push_back(C[i] * v.norm());
  }
  double b_sum = std::accumulate(b.begin(), b.end(), 0.0);

  observer->OnStart();
  stopping_rule->Start();
  for (int it = 0; stopping_rule->ShouldContinue(it); it++) {
    if (it % 1000 == 0) std::cout << "iteration " << it << "\r";
    // Compute the dominating rate for each coordinate (denoted M_i(t) in the
    // Zig-Zag paper).
    // These bounds are of the form a + b t.
    vector<double> a;
    for (int i = 0; i < d; i++) {
      a.push_back(std::max(0.0, global_gradient_at_mode[i] * v[i])
                  + C[i] * (theta - mode).norm());
    }

    double a_sum = std::accumulate(a.begin(), a.end(), 0.0);

    // Simulate a candidate time with rate sum_bounds.
    double U = gsl_ran_exponential(rng, 1.0);
    double candidate_time = quadratic_solution(0.5 * b_sum, a_sum, -U);

    // Select an individual coordinate k.
    double u = (a_sum + candidate_time * b_sum) * gsl_rng_uniform(rng);
    int k = 0;
    while (u > a[k] + candidate_time * b[k]) {
      u -= a[k] + candidate_time * b[k];
      k++;
    }

    // Compute the bounding rate as it applies to coordinate k.
    double bounding_rate = a[k] + candidate_time * b[k];
    bounding_rate /= N;

    // Pick a random data point.
    int J = gsl_rng_uniform_int(rng, N);

    // Thin the coordinate time.
    // Compute the actual rate m_i at the candidate time.
    double actual_rate = v[k] * (global_gradient_at_mode[k] / N
                                 + target->GradientCoordinate(J, k, theta + candidate_time * v)
                                 - target_gradients_at_mode[J][k]);

    assert(actual_rate < bounding_rate);

    // Record the duration of the current state before moving.
    observer->OnSample(theta, v, candidate_time);

    // Advance forward and bounce if accepted.
    theta += candidate_time * v;
    if (actual_rate > 0 && gsl_rng_uniform(rng) < actual_rate / bounding_rate) {
      v[k] = -v[k];
    }
  }
  observer->OnComplete();
}
