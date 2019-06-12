#pragma once

#include <Eigen/Core>
#include <gsl/gsl_rng.h>

#include "lr_target.h"
#include "observer.h"
#include "stopping.h"

void lr_zigzag(gsl_rng* rng,
               const LrTarget* target,
               const Eigen::VectorXd& mode,
               const Eigen::VectorXd& theta0,
               StoppingRule* stopping_rule,
               ContinuousTimeObserver* observer);
