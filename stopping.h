#pragma once

#include <boost/timer/timer.hpp>

class StoppingRule {
 public:
  virtual void Start() = 0; 
  virtual bool ShouldContinue(int iteration) = 0;
};

class IterationsStoppingRule : public StoppingRule {
 public:
  IterationsStoppingRule(int min_iterations, int min_seconds)
    : min_iterations_(min_iterations),
      min_nanoseconds_(min_seconds * 1000000000LL),
      started_(false)
  {}
  
  virtual void Start() {
    started_ = true;
    start_time_ = timer_.elapsed();
  }
  
  virtual bool ShouldContinue(int iteration) {
    if (iteration < min_iterations_) {
      return true;
    }
    assert(started_);
    if (iteration % 1000 == 0) {
      boost::timer::cpu_times current_time = timer_.elapsed();
      if (current_time.system + current_time.user
          - start_time_.system - start_time_.user < min_nanoseconds_) {
        return true;
      } else {
        return false;
      }
    }
    return true;
  }
   
 private:
  int min_iterations_;
  boost::timer::cpu_timer timer_;
  boost::timer::cpu_times start_time_;
  boost::timer::nanosecond_type min_nanoseconds_;
  bool started_;
};
