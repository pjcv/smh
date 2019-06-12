#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE alias test

#include <boost/test/unit_test.hpp>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "alias.h"

using std::unordered_map;
using std::vector;

BOOST_AUTO_TEST_CASE(alias_test) {
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, 8);

  int N = 10;
  vector<double> probs;
  for (int i = 0; i < N; i++) {
    probs.push_back(gsl_rng_uniform(rng));
  }
  double prob_sum = std::accumulate(probs.begin(), probs.end(), 0.0);

  for (double& prob : probs) {
    prob /= prob_sum;
  }

  Alias a(rng, probs);

  vector<int> samples = a.Sample(10000);

  // TODO: a statistical test would be appropriate here.
  std::unordered_map<int, int> hist;
  for (int sample : samples) {
    hist[sample] += 1;
  }

  for (int i = 0; i < N; i++) {
    std::cout << i << " " << probs[i] << " " << double(hist[i]) / samples.size() << std::endl;
  }
}

