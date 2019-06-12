#pragma once

#include <Eigen/Dense>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <json/json.h>
#include <memory>

#include "target.h"
#include "util.h"

class Proposal {
 public:
  // Samples a proposal, which may depend on the current state.
  // Store in potential_difference_out the negative log density of the density
  // ratio as seen in the MH algorithm, i.e.
  // - log [ q(theta | theta') / q(theta' | theta) ]
  virtual void Sample(const Eigen::VectorXd& current_theta,
                      Eigen::VectorXd* proposed_theta_out,
                      double* potential_difference_out) = 0;
};

template <typename MatT>
class RandomWalkProposal : public Proposal {
 public:
  RandomWalkProposal(gsl_rng* rng, const MatT& cholesky)
    : rng_(rng),
      cholesky_(cholesky)
  {}

  virtual void Sample(const Eigen::VectorXd& current_theta,
                      Eigen::VectorXd* proposed_theta_out,
                      double* potential_difference_out) {
    *proposed_theta_out = current_theta + cholesky_ * GaussianVector(rng_, current_theta.size());
    *potential_difference_out = 0.0;
  }

 private:
  gsl_rng* rng_;
  MatT cholesky_;
};

class PcnProposal : public Proposal {
 public:
  PcnProposal(gsl_rng* rng,
              double rho,
              const Eigen::VectorXd& theta_hat,
              const Target* target)
    : rng_(rng)
  {
    int d = target->Dimension();
    A_ = sqrt(rho) * Eigen::MatrixXd::Identity(d, d);
    Eigen::MatrixXd hessian_inv = target->Hessian(theta_hat).inverse();
    b_ = (1 - sqrt(rho)) * (theta_hat - hessian_inv * target->Gradient(theta_hat));
    Eigen::MatrixXd C = (1 - rho) * hessian_inv;
    cholesky_C_ = C.llt().matrixL();
    inv_C_ = target->Hessian(theta_hat) / (1 - rho);
  }

  virtual void Sample(const Eigen::VectorXd& current_theta,
                      Eigen::VectorXd* proposed_theta_out,
                      double* potential_difference_out) {
    Eigen::VectorXd proposal_mean = A_ * current_theta + b_;
    Eigen::VectorXd proposal_noise = cholesky_C_ * GaussianVector(rng_, current_theta.size());
    *proposed_theta_out = proposal_mean + proposal_noise;

    double proposal_potential = 0.5 * proposal_noise.dot(inv_C_ * proposal_noise);
    Eigen::VectorXd reverse_proposal_mean = A_ * *proposed_theta_out + b_;
    double reverse_proposal_potential = 0.5 * (current_theta - reverse_proposal_mean).dot(inv_C_ * (current_theta - reverse_proposal_mean));
    *potential_difference_out = reverse_proposal_potential - proposal_potential;
  }

 private:
  gsl_rng* rng_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::MatrixXd cholesky_C_;
  Eigen::MatrixXd inv_C_;
};
