#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <H5Cpp.h>
#include <iomanip>
#include <iostream>
#include <json/json.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "io.h"
#include "target.h"

namespace po = boost::program_options;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using H5::DataSet;
using H5::DataSpace;
using H5::H5File;
using H5::PredType;
using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char** argv) {
  Json::Value root_config = cfg::load_config(argc, argv);

  assert(root_config.isMember("output_filename"));
  std::string output_filename = root_config.get("output_filename", "").asString();

  std::unique_ptr<Target> target = cfg::load_target(root_config);
  int d = target->Dimension();
  std::cout << "read dataset with d=" << target->Dimension() << " N=" << target->Count() << std::endl;

  VectorXd theta_star = VectorXd::Zero(d);
  int optimization_iterations = 0;
  {
    boost::timer::auto_cpu_timer t("Optimization: %u user, %s system, %w wall\n");
    double alpha = 1.0;
    double current_potential = target->Potential(theta_star);

    std::cout << "initial potential: " << current_potential << std::endl;

    while (true) {
      optimization_iterations++;
      VectorXd grad = target->Gradient(theta_star);
      // TODO: should we condition on the grad norm instead?
      if ((alpha * grad).squaredNorm() < 1e-10) {
        break;
      }
      VectorXd theta_prop = theta_star - alpha * grad;
      double prop_potential = target->Potential(theta_prop);
      while (prop_potential > current_potential) {
        alpha *= 0.5;
        theta_prop = theta_star - alpha * grad;
        prop_potential = target->Potential(theta_prop);
      }
      theta_star = theta_prop;
      current_potential = prop_potential;
      alpha *= 1.5;

      if (!isfinite(current_potential)) {
        std::cerr << "current_potential " << current_potential << ", exiting..." << std::endl;
        return -1;
      }

      if (optimization_iterations > 1000000) {
        std::cerr << "exiting after " << optimization_iterations << " iterations" << std::endl;
        break;
      }
    }
  }

  MatrixXd hessian = target->Hessian(theta_star);
  MatrixXd covariance = hessian.inverse();

  std::cout << "writing " << output_filename << " ..." << std::endl;
  H5File file(output_filename, H5F_ACC_TRUNC);
  write_args(&file, "args", argc, argv);
  write_config(&file, root_config);

  write_vector(&file, "mode", theta_star);
  write_matrix(&file, "covariance", covariance);
}
