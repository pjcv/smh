#include <Eigen/Dense>
#include <H5Cpp.h>
#include <json/json.h>

#include "acf.h"
#include "config.h"
#include "io.h"

using std::string;

int main(int argc, char** argv) {
  Json::Value root = cfg::load_config(argc, argv);

  assert(root.isMember("output_filename"));
  string output_filename = root.get("output_filename", "").asString();

  assert(root.isMember("samples_filename"));
  std::string samples_filename = root.get("samples_filename", "").asString();

  H5::H5File fh(samples_filename, H5F_ACC_RDONLY);

  Eigen::VectorXd samples = read_vector(&fh, "samples");
  double iact = iact_overlapping_batch_means(samples);

  double iterations = samples.size();
  double seconds = static_cast<double>(read_int64(&fh, "nanoseconds")) / 1e9;
  double likelihood_evaluations = read_int64(&fh, "likelihood_evaluations");

  H5::H5File ofh(output_filename, H5F_ACC_TRUNC);
  write_args(&ofh, "args", argc, argv);
  write_config(&ofh, root);
  write_double(&ofh, "iact_iterations", iact / iterations);
  write_double(&ofh, "iact_seconds", iact / iterations * seconds);
  write_double(&ofh, "iact_likelihood_evaluations", iact / iterations * likelihood_evaluations);
}
