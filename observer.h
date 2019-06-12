#pragma once

#include <boost/timer/timer.hpp>
#include <Eigen/Core>
#include <H5Cpp.h>
#include <memory>
#include <vector>
#include <unordered_map>

#include "target.h"


class Observer {
 public:
  virtual void OnStart() = 0;
  virtual void OnSample(const Eigen::VectorXd& sample) = 0;
  virtual void OnComplete() = 0;
};

class SampleRecorder : public Observer {
 public:
  SampleRecorder()
  {}

  void OnStart() {
  }
  
  void OnSample(const Eigen::VectorXd& s) {
    samples.push_back(s[0]);
  }

  void OnComplete() {
  }

  void Serialize(H5::H5File* file) {
    std::unique_ptr<double[]> data(new double[samples.size()]);
    for (int i = 0; i < samples.size(); i++) {
      data[i] = samples[i];
    }
    int rank = 1;
    hsize_t dims[rank];
    dims[0] = samples.size();
    H5::DataSpace dataspace(rank, dims);
    H5::DataSet dataset = file->createDataSet("samples", H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
  }

  std::vector<double> samples;
};


class ContinuousTimeObserver {
 public:
  virtual void OnStart() = 0;
  virtual void OnSample(const Eigen::VectorXd& x,
                        const Eigen::VectorXd& v,
                        double duration) = 0;
  virtual void OnComplete() = 0;
};

class ContinuousTimeSampleRecorder : public ContinuousTimeObserver {
 public:
  ContinuousTimeSampleRecorder() {
    start_x_ = NAN;
    previous_v_ = NAN;
    accumulated_duration_ = 0;
  }
  
  void OnStart() {}
  
  void OnSample(const Eigen::VectorXd& x, const Eigen::VectorXd& v, double duration) {
    if (v[0] == previous_v_) {
      accumulated_duration_ += duration;
    } else {     
      if (!std::isnan(start_x_)) {
        x_samples.push_back(start_x_);
        v_samples.push_back(previous_v_);
        durations.push_back(accumulated_duration_);
      }
      start_x_ = x[0];
      previous_v_ = v[0];
      accumulated_duration_ = duration;
    }
  }

  void OnComplete() {
    x_samples.push_back(start_x_);
    v_samples.push_back(previous_v_);
    durations.push_back(accumulated_duration_);
  }

  void Serialize(H5::H5File* file) {
     {
       std::unique_ptr<double[]> data(new double[x_samples.size()]);
       for (int i = 0; i < x_samples.size(); i++) {
         data[i] = x_samples[i];
       }
       int rank = 1;
       hsize_t dims[rank];
       dims[0] = x_samples.size();
       H5::DataSpace dataspace(rank, dims);
       H5::DataSet dataset = file->createDataSet("x_samples", H5::PredType::NATIVE_DOUBLE, dataspace);
       dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
     }
     
     {
       std::unique_ptr<double[]> data(new double[v_samples.size()]);
       for (int i = 0; i < v_samples.size(); i++) {
         data[i] = v_samples[i];
       }
       int rank = 1;
       hsize_t dims[rank];
       dims[0] = v_samples.size();
       H5::DataSpace dataspace(rank, dims);
       H5::DataSet dataset = file->createDataSet("v_samples", H5::PredType::NATIVE_DOUBLE, dataspace);
       dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
     }
     {
       std::unique_ptr<double[]> data(new double[durations.size()]);
       for (int i = 0; i < durations.size(); i++) {
         data[i] = durations[i];
       }
       int rank = 1;
       hsize_t dims[rank];
       dims[0] = v_samples.size();
       H5::DataSpace dataspace(rank, dims);
       H5::DataSet dataset = file->createDataSet("durations", H5::PredType::NATIVE_DOUBLE, dataspace);
       dataset.write(data.get(), H5::PredType::NATIVE_DOUBLE);
     }
  }

  std::vector<double> x_samples;
  std::vector<double> v_samples;
  std::vector<double> durations;
  
 private:
  double start_x_;
  double previous_v_;
  double accumulated_duration_;
};

