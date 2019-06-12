#pragma once

#include <Eigen/Core>
#include <gsl/gsl_rng.h>

#include "observer.h"
#include "proposal.h"
#include "stopping.h"


void fmh(gsl_rng* rng,
         const Target* target,
         const Eigen::VectorXd& mode,
         int taylor_order,
         Proposal* proposal,
         const Eigen::VectorXd& theta0,
         StoppingRule* stopping_rule,
         Observer* observer);
