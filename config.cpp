#include "config.h"

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "io.h"
#include "lr_target.h"
#include "rlr_target.h"

namespace cfg {

namespace po = boost::program_options;

using std::string;

Json::Value load_config(int argc, char** argv) {
  po::options_description desc("options");
  desc.add_options()
    ("config", po::value<string>()->required(), "")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  string config = vm["config"].as<string>();

  std::ifstream test(config, std::ifstream::binary);
  if (!test.is_open()) {
    std::cerr << "could not open " << config << std::endl;
    assert(false);
  }

  Json::CharReaderBuilder builder;
  Json::Value root;
  string errs;
  if (!Json::parseFromStream(builder, test, &root, &errs)) {
    std::cerr << "json parsing error:" << std::endl;
    std::cerr << errs << std::endl;
    assert(false);
  }

  return root;
}

Eigen::VectorXd load_vector(const Json::Value& config) {
  assert(config.get("type", "").asString() == "hdf5");

  string path = config.get("file", "").asString();
  string dataset = config.get("dataset", "").asString();

  H5::H5File fh(path, H5F_ACC_RDONLY);
  return read_vector(&fh, dataset);
}

Eigen::MatrixXd load_matrix(const Json::Value& config) {
  assert(config.get("type", "").asString() == "hdf5");

  string path = config.get("file", "").asString();
  string dataset = config.get("dataset", "").asString();
  
  H5::H5File fh(path, H5F_ACC_RDONLY);
  return read_matrix(&fh, dataset);
}

std::unique_ptr<Target> load_target(const Json::Value& config) {

  assert(config.isMember("dataset_filename"));
  std::string file = config.get("dataset_filename", "").asString();

  assert(config.isMember("model"));
  Json::Value model = config.get("model", Json::Value::null);

  assert(model.isMember("type"));
  std::string model_type = model.get("type", "").asString();

  H5::H5File fh(file, H5F_ACC_RDONLY);

  Eigen::VectorXd y = read_vector(&fh, "y");
  Eigen::MatrixXd X = read_matrix(&fh, "X");

  std::unique_ptr<Target> rv;
  if (model_type == "lr") {
    rv = std::make_unique<LrTarget>(y, X);
  } else if (model_type == "rlr") {
    assert(model.isMember("params"));
    Json::Value model_params = model.get("params", Json::Value::null);
    assert(model_params.isMember("nu"));
    double nu = model_params.get("nu", 0.0).asDouble();
    rv = std::make_unique<RlrTarget>(y, X, nu);
  }
  return rv;
}
  
}  // namespace cfg
