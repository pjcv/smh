#include "acf.h"

using Eigen::ArrayXd;
using Eigen::VectorXd;

double iact_overlapping_batch_means(const VectorXd& seq) {
  int N = seq.size();
  ArrayXd seq0 = seq.array() - seq.mean();

  double var = 0.0;
  for (int i = 0; i < N; i++) {
    var += seq0[i] * seq0[i];
  }
  var /= N;

  int batch_length = pow(N, 2.0 / 3.0);
  int num_batches = N - batch_length + 1;

  ArrayXd cs = ArrayXd::Zero(N + 1);
  for (int i = 0; i < N; i++) {
    cs[i + 1] = cs[i] + seq0[i];
  }

  ArrayXd diffs = ArrayXd::Zero(num_batches);
  for (int i = 0; i < num_batches; i++) {
    diffs[i] = cs[i + batch_length] - cs[i];
  }
  ArrayXd batch_means = diffs / batch_length;

  double sigma2 = (batch_means * batch_means).sum() * N * batch_length
    / (N - batch_length)
    / (N - batch_length + 1);

  return sigma2 / var;
}
