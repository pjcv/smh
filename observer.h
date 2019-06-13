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
    start_time_ = timer_.elapsed();
  }
  
  void OnSample(const Eigen::VectorXd& s) {
    samples.push_back(s[0]);
  }

  void OnComplete() {
    boost::timer::cpu_times end_time(timer_.elapsed());
    elapsed_nanoseconds = end_time.system + end_time.user - start_time_.system - start_time_.user;
  }

  std::vector<double> samples;
  int64_t elapsed_nanoseconds;  

 private:
  boost::timer::cpu_times start_time_;
  boost::timer::cpu_timer timer_;
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
  
  void OnStart() {
    start_time_ = timer_.elapsed();
  }
  
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
    boost::timer::cpu_times end_time(timer_.elapsed());
    elapsed_nanoseconds = end_time.system + end_time.user - start_time_.system - start_time_.user;
  }

  std::vector<double> x_samples;
  std::vector<double> v_samples;
  std::vector<double> durations;
  int64_t elapsed_nanoseconds;  
  
 private:
  double start_x_;
  double previous_v_;
  double accumulated_duration_;
  boost::timer::cpu_times start_time_;
  boost::timer::cpu_timer timer_;
};

