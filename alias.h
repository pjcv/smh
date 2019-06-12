#pragma once

#include <gsl/gsl_rng.h>
#include <numeric>
#include <vector>

class Alias {
 public:
  Alias(gsl_rng* rng, const std::vector<double>& probs)
    : rng_(rng)
  {
    int K = probs.size();
    double sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0);

    for (int i = 0; i < K; i++) {
      q_.push_back(K * probs[i] / sum_probs);
      j_.push_back(i);
    }

    std::vector<double> g;
    std::vector<double> s;
    for (int i = 0; i < K; i++) {
      if (q_[i] > 1.) {
	g.push_back(i);
      } else {
	s.push_back(i);
      }
    }

    while (g.size() > 0 && s.size() > 0) {
      int k = g.back(); 
      int l = s.back();
      s.pop_back();

      j_[l] = k;
      q_[k] +=  q_[l] - 1.0;

      if (q_[k] < 1.0) {
	g.pop_back();
	s.push_back(k);
      }
    }
  }

  int Sample() {
    std::vector<int> rv;
    int X = gsl_rng_uniform_int(rng_, q_.size());
    double V = gsl_rng_uniform(rng_);
    if (V < q_[X]) {
      return X;
    } else {
      return j_[X];
    }
  }
  
  std::vector<int> Sample(int N) {
    std::vector<int> rv;
    for (int i = 0; i < N; i++) {
      rv.push_back(Sample());
    }
    return rv;
  }

 private:
  gsl_rng* rng_;
  std::vector<double> q_;
  std::vector<int> j_;
};
