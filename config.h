#pragma once

#include <Eigen/Core>
#include <json/json.h>
#include <memory>

#include "target.h"

namespace cfg {

// Load json file given by the --config command-line argument. 
Json::Value load_config(int argc, char** argv);

// Load Eigen objects from a "standard" json description.
Eigen::VectorXd load_vector(const Json::Value& config);
Eigen::MatrixXd load_matrix(const Json::Value& config);

// Load Target object given root config.
std::unique_ptr<Target> load_target(const Json::Value& config);

}
