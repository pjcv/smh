#include <Eigen/Dense>
#include <gsl/gsl_randist.h>
#include <H5Cpp.h>
#include <json/json.h>

#include "config.h"
#include "io.h"
#include "target.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;

int main(int argc, char** argv) {
  Json::Value root_config = cfg::load_config(argc, argv);

  assert(root_config.isMember("output_filename"));
  string output_filename = root_config.get("output_filename", "").asString();

  assert(root_config.isMember("covariate_distribution"));
  string distribution = root_config.get("covariate_distribution", "").asString();

  assert(root_config.isMember("N"));
  int N = root_config.get("N", 0).asInt();

  assert(root_config.isMember("theta"));
  Json::Value theta_value = root_config.get("theta", Json::Value::null);
  int d = theta_value.size();
  VectorXd theta_vector(d);
  for (int i = 0; i < d; i++) {
    theta_vector[i] = theta_value[i].asDouble();
  }

  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);

  assert(root_config.isMember("seed"));
  gsl_rng_set(rng, root_config.get("seed", 0).asInt());

  assert(root_config.isMember("noise"));
  string noise = root_config.get("noise", "").asString();
  assert(noise == "gaussian" || noise == "bernoulli");

  // Generate data.
  VectorXd y = VectorXd(N);
  CovariateT X = CovariateT(N, d);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < d; j++) {
      if (distribution == "gaussian") {
        X(i, j) = gsl_ran_gaussian(rng, 1.0);
      } else if (distribution == "cauchy") {
        X(i, j) = gsl_ran_cauchy(rng, 1.0);
      } else if (distribution == "laplace") {
        X(i, j) = gsl_ran_laplace(rng, 1.0);
      } else if (distribution == "uniform") {
        X(i, j) = gsl_rng_uniform(rng) - 0.5;
      }
    }

    double z = X.row(i).dot(theta_vector);

    if (noise == "gaussian") {
      y[i] = z + gsl_ran_gaussian(rng, 1.0);
    } else if (noise == "bernoulli") {
      double prob = exp(z) / (exp(z) + 1.0);
      if (gsl_rng_uniform(rng) < prob) {
        y[i] = 1.0;
      } else {
        y[i] = -1.0;
      }
    }
  }

  H5::H5File ofh(output_filename, H5F_ACC_TRUNC);
  write_args(&ofh, "args", argc, argv);
  write_config(&ofh, root_config);
  write_vector(&ofh, "y", y);
  write_matrix(&ofh, "X", X);
}
