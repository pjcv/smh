#include <boost/timer/timer.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <gsl/gsl_randist.h>
#include <H5Cpp.h>
#include <iostream>
#include <vector>

#include "acf.h"
#include "config.h"
#include "io.h"
#include "lr_zigzag.h"
#include "util.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char** argv) {
  Json::Value root_config = cfg::load_config(argc, argv);

  assert(root_config.isMember("output_filename"));
  string output_filename = root_config.get("output_filename", "").asString();

  assert(root_config.isMember("model"));
  Json::Value model = root_config.get("model", Json::Value::null);
  assert(model.get("type", "").asString() == "lr");

  assert(root_config.isMember("dataset_filename"));
  std::string dataset_filename = root_config.get("dataset_filename", "").asString();

  H5::H5File fh(dataset_filename, H5F_ACC_RDONLY);
  Eigen::VectorXd y = read_vector(&fh, "y");
  Eigen::MatrixXd X = read_matrix(&fh, "X");

  LrTarget target(y, X);

  assert(root_config.isMember("iterations"));
  int iterations = root_config.get("iterations", 0).asInt();

  unique_ptr<StoppingRule> stopping_rule = std::make_unique<IterationsStoppingRule>(iterations, 1.0);
  
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  assert(root_config.isMember("seed"));
  int seed = root_config.get("seed", 0).asInt();
  gsl_rng_set(rng, seed);

  ContinuousTimeSampleRecorder recorder = ContinuousTimeSampleRecorder();

  VectorXd theta0 = cfg::load_vector(root_config.get("initial_theta", Json::Value::null));
  VectorXd mode = cfg::load_vector(root_config.get("mode", Json::Value::null));

  boost::timer::cpu_timer timer;
  boost::timer::cpu_times start_time;
  boost::timer::cpu_times end_time;

  start_time = timer.elapsed();
  lr_zigzag(rng,
            &target,
            mode,
            theta0,
            stopping_rule.get(),
            &recorder);
  end_time = timer.elapsed();

  // Write data.
  H5::H5File ofh(output_filename, H5F_ACC_TRUNC);
  write_args(&ofh, "args", argc, argv);
  write_config(&ofh, root_config);

  vector<double> discrete_samples;
  subsample_path(recorder.x_samples,
                 recorder.v_samples,
                 recorder.durations,
                 recorder.x_samples.size() * 100,
                 &discrete_samples);

  write_doubles(&ofh, "samples", discrete_samples);

  int64_t num_potential_evaluations = target.NumPotentialEvaluations();
  write_int64(&ofh, "likelihood_evaluations", num_potential_evaluations);

  int64_t elapsed_nanoseconds = end_time.system + end_time.user - start_time.system - start_time.user;
  write_int64(&ofh, "nanoseconds", elapsed_nanoseconds);
}
