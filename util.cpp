#include <util.h>

#include <numeric>

using std::vector;

void subsample_path(const vector<double>& x,
                    const vector<double>& v,
                    const vector<double>& t,
                    int num_samples,
                    vector<double>* x_samples_out) {
  assert(x.size() == v.size());
  assert(x.size() == t.size());

  double T = std::accumulate(t.begin(), t.end(), 0.0);
  double stride = T / t.size();

  double segment_t = 0.0;
  x_samples_out->clear();
  for (int i = 0;  i < x.size(); i++) {
    while (segment_t < t[i]) {
      x_samples_out->push_back(x[i] + v[i] * segment_t);
      segment_t += stride;
    }
    segment_t -= t[i];
  }
}
