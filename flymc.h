#pragma once

#include <Eigen/Core>
#include <gsl/gsl_rng.h>

#include "observer.h"
#include "proposal.h"
#include "stopping.h"
#include "target.h"


void flymc(gsl_rng* rng,
           const Target* target,
           const Eigen::VectorXd& mode,
           Proposal* proposal,
           double qdb,
           const Eigen::VectorXd& theta0,
           StoppingRule* stopping_rule,
           Observer* observer);
