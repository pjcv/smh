#pragma once

#include <gsl/gsl_rng.h>
#include <Eigen/Core>

#include "observer.h"
#include "proposal.h"
#include "stopping.h"
#include "target.h"

void mh(gsl_rng* rng,
        const Target* target,
        Proposal* proposal,
        const Eigen::VectorXd& theta0,
        StoppingRule* stopping_rule,
        Observer* observer);
