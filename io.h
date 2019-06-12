#pragma once

#include <Eigen/Core>
#include <H5Cpp.h>
#include <json/json.h>
#include <string>
#include <vector>

// Write out args to HDF5 file as one-dimensional array of strings.
void write_args(H5::H5File* file,
                const std::string& dataset_path,
                int argc,
                char** argv);

void write_string(H5::H5File* file,
                  const std::string& dataset_path,
                  const std::string& data);

void write_config(H5::H5File* file,
                  const Json::Value& config);

void write_double(H5::H5File* file,
                 const std::string& dataset_path,
                 double x);

void write_int64(H5::H5File* file,
                 const std::string& dataset_path,
                 int64_t x);

void write_doubles(H5::H5File* file,
                   const std::string& dataset_path,
                   const std::vector<double>& xs);

void write_vector(H5::H5File* file,
                  const std::string& dataset_path,
                  const Eigen::VectorXd& vector);

void write_matrix(H5::H5File* file,
                  const std::string& dataset_path,
                  const Eigen::MatrixXd& matrix);

int64_t read_int64(H5::H5File* file,
                   const std::string& dataset_path);

Eigen::VectorXd read_vector(H5::H5File* file,
                            const std::string& dataset_path);

Eigen::MatrixXd read_matrix(H5::H5File* file,
                            const std::string& dataset_path);
