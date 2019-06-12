#include "io.h"

#include <cstring>
#include <Eigen/Core>
#include <H5Cpp.h>
#include <memory>
#include <sstream>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using H5::DataSet;
using H5::DataSpace;
using H5::H5File;
using H5::PredType;
using std::string;
using std::unique_ptr;

void write_args(H5File* file, const string& dataset_name, int argc, char** argv) {
  unique_ptr<char*[]> raw_data(new char*[argc]);
  for (int i = 0; i < argc; i++) {
    int buf_len = strlen(argv[i]) + 1;
    raw_data[i] = new char[buf_len];
    strncpy(raw_data[i], argv[i], buf_len);
  }

  hsize_t dims[1] = {(hsize_t)argc};
  int rank = 1;
  DataSpace dataspace(rank, dims);

  hid_t stringtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(stringtype, H5T_VARIABLE);

  DataSet dataset = file->createDataSet(dataset_name, stringtype, dataspace);
  dataset.write(raw_data.get(), stringtype);

  for (int i = 0; i < argc; i++) {
    delete raw_data[i];
  }
}

void write_string(H5File* file, const string& dataset_name, const string& data) {
  hsize_t dims[1] = {(hsize_t)1};
  int rank = 1;
  DataSpace dataspace(rank, dims);

  hid_t stringtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(stringtype, H5T_VARIABLE);

  DataSet dataset = file->createDataSet(dataset_name, stringtype, dataspace);

  unique_ptr<char[]> c_str(new char[data.size() + 1]);
  strncpy(c_str.get(), data.c_str(), data.size() + 1);
  char* c_str_array = {c_str.get()};
  dataset.write(&c_str_array, stringtype);
}

void write_config(H5::H5File* file, const Json::Value& config) {
  Json::StreamWriterBuilder builder;
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  std::stringstream sstream;
  writer->write(config, &sstream);
  write_string(file, "config", sstream.str());
}

void write_double(H5::H5File* file,
                 const std::string& dataset_path,
                 double x) {
  int rank = 1;
  hsize_t dims[1];
  dims[0] = 1;
  H5::DataSpace dataspace(rank, dims);
  H5::DataSet dataset = file->createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(&x, H5::PredType::NATIVE_DOUBLE);
}

void write_doubles(H5::H5File* file,
                   const std::string& dataset_path,
                   const std::vector<double>& xs) {
  std::unique_ptr<double[]> data(new double[xs.size()]);
  for (int i = 0; i < xs.size(); i++) {
    data[i] = xs[i];
  }
  int rank = 1;
  hsize_t dims[1];
  dims[0] = xs.size();
  H5::DataSpace dataspace(rank, dims);
  H5::DataSet dataset = file->createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
}


void write_int64(H5::H5File* file,
                 const std::string& dataset_path,
                 int64_t x) {
  int rank = 1;
  hsize_t dims[1];
  dims[0] = 1;
  H5::DataSpace dataspace(rank, dims);
  H5::DataSet dataset = file->createDataSet(dataset_path, H5::PredType::NATIVE_INT64, dataspace);
  dataset.write(&x, H5::PredType::NATIVE_INT64);
}

void write_vector(H5::H5File* file,
                  const std::string& dataset_path,
                  const Eigen::VectorXd& vector) {
  unique_ptr<double[]> data(new double[vector.size()]);
  for (int i = 0; i < vector.size(); i++) {
    data[i] = vector[i];
  }
  int rank = 1;
  hsize_t dims[1];
  dims[0] = vector.size();
  H5::DataSpace dataspace(rank, dims);
  H5::DataSet dataset = file->createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
}

void write_matrix(H5::H5File* file,
                  const std::string& dataset_path,
                  const Eigen::MatrixXd& matrix) {
  unique_ptr<double[]> data(new double[matrix.rows() * matrix.cols()]);
  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < matrix.cols(); j++) {
      data[i * matrix.cols() + j] = matrix(i, j);
    }
  }
  int rank = 2;
  hsize_t dims[2];
  dims[0] = matrix.rows();
  dims[1] = matrix.cols();
  H5::DataSpace dataspace(rank, dims);
  H5::DataSet dataset = file->createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
}

int64_t read_int64(H5::H5File* file,
                   const std::string& dataset_name) {
  DataSet dataset = file->openDataSet(dataset_name);
  DataSpace dataspace = dataset.getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  assert(rank == 1);
  hsize_t dims_out[1];
  int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
  assert(dims_out[0] == 1);
  int64_t result;
  dataset.read(&result, PredType::NATIVE_INT64);
  return result;
}

Eigen::VectorXd read_vector(H5::H5File* file,
                            const std::string& dataset_name) {
  DataSet dataset = file->openDataSet(dataset_name);
  DataSpace dataspace = dataset.getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  assert(rank == 1);
  hsize_t dims_out[1];
  int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
  unique_ptr<double[]> raw_data(new double[dims_out[0]]);
  dataset.read(raw_data.get(), PredType::NATIVE_DOUBLE);
  VectorXd result = VectorXd(dims_out[0]);
  for (int i = 0; i < dims_out[0]; i++) {
    result(i) = raw_data[i];
  }
  return result;
}

Eigen::MatrixXd read_matrix(H5::H5File* file,
                            const std::string& dataset_name) {
  DataSet dataset = file->openDataSet(dataset_name);
  DataSpace dataspace = dataset.getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  assert(rank == 2);
  hsize_t dims_out[2];
  int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
  std::unique_ptr<double[]> raw_data(new double[dims_out[0] * dims_out[1]]);
  dataset.read(raw_data.get(), PredType::NATIVE_DOUBLE);
  MatrixXd result = MatrixXd(dims_out[0], dims_out[1]);
  for (int i = 0; i < dims_out[0]; i++) {
    for (int j = 0; j < dims_out[1]; j++) {
      result(i, j) = raw_data[i * dims_out[1] + j];
    }
  }
  return result;
}
