#include <boost/timer/timer.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <gsl/gsl_randist.h>
#include <H5Cpp.h>
#include <iomanip>
#include <iostream>
#include <json/json.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "flymc.h"
#include "fmh.h"
#include "io.h"
#include "mh.h"
#include "proposal.h"
#include "target.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::make_unique;
using std::string;
using std::unique_ptr;
using std::vector;


int main(int argc, char** argv) {
  Json::Value root = cfg::load_config(argc, argv);

  assert(root.isMember("output_filename"));
  string output_filename = root.get("output_filename", "").asString();

  std::unique_ptr<Target> target = cfg::load_target(root);

  assert(root.isMember("alg"));
  Json::Value alg = root.get("alg", Json::Value::null);
  assert(alg.isMember("type"));
  string alg_type = alg.get("type", "").asString();

  unique_ptr<StoppingRule> stopping_rule = make_unique<IterationsStoppingRule>(root.get("iterations", 0).asInt(), 1.0);

  assert(root.isMember("seed"));
  int seed = root.get("seed", 0).asInt();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
  gsl_rng_set(rng, seed);

  SampleRecorder recorder;

  VectorXd theta0 = cfg::load_vector(root.get("initial_theta", Json::Value::null));
  VectorXd mode = cfg::load_vector(root.get("mode", Json::Value::null));

  assert(root.isMember("proposal"));
  Json::Value proposal_config = root.get("proposal", Json::Value::null);

  assert(proposal_config.isMember("type"));
  string proposal_type = proposal_config.get("type", "").asString();

  unique_ptr<Proposal> proposal;
  if (proposal_type == "random_walk") {
    MatrixXd hessian = target->Hessian(mode);
    MatrixXd mass_chol = hessian.inverse().llt().matrixL();
    proposal = make_unique<RandomWalkProposal<MatrixXd>>(rng, mass_chol);
  } else if (proposal_type == "pcn") {
    assert(proposal_config.isMember("params"));
    Json::Value proposal_params = proposal_config.get("params", Json::Value::null);
    assert(proposal_params.isMember("rho"));
    double rho = proposal_params.get("rho", 0.0).asDouble();
    proposal = make_unique<PcnProposal>(rng, rho, mode, target.get());
  } else {
    std::cerr << "unrecognized proposal type: " << proposal_type << std::endl;
    assert(false);
  }

  boost::timer::cpu_timer timer;
  boost::timer::cpu_times start_time;
  boost::timer::cpu_times end_time;

  if (alg_type == "fmh") {
    Json::Value alg_params = alg.get("params", Json::Value::null);
    assert(alg_params.isMember("taylor_order"));
    int taylor_order = alg_params.get("taylor_order", 1).asInt();

    start_time = timer.elapsed();
    fmh(rng,
        target.get(),
        mode,
        taylor_order,
        proposal.get(),
        theta0,
        stopping_rule.get(),
        &recorder);
    end_time = timer.elapsed();

  } else if (alg_type == "mh") {
    start_time = timer.elapsed();
    mh(rng,
       target.get(),
       proposal.get(),
       theta0,
       stopping_rule.get(),
       &recorder);
    end_time = timer.elapsed();

  } else if (alg_type == "flymc") {
    Json::Value alg_params = alg.get("params", Json::Value::null);
    assert(alg_params.isMember("qdb"));
    double qdb = alg_params.get("qdb", 0.0).asDouble();

    start_time = timer.elapsed();
    flymc(rng,
          target.get(),
          mode,
          proposal.get(),
          qdb,
          theta0,
          stopping_rule.get(),
          &recorder);
    end_time = timer.elapsed();

  } else {
    std::cerr << "unrecognized alg: " << alg << std::endl;
    return -1;
  }

  H5::H5File ofh(output_filename, H5F_ACC_TRUNC);
  write_args(&ofh, "args", argc, argv);
  write_config(&ofh, root);
  recorder.Serialize(&ofh);

  int64_t num_potential_evaluations = target->NumPotentialEvaluations();
  write_int64(&ofh, "likelihood_evaluations", num_potential_evaluations);

  int64_t elapsed_nanoseconds = end_time.system + end_time.user - start_time.system - start_time.user;
  write_int64(&ofh, "nanoseconds", elapsed_nanoseconds);
}
