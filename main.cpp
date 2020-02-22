#include <iostream>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  device.setup((std::string) args["options/device"]);

  // device.setup("mode: 'Serial'");

  // device.setup("mode     : 'OpenMP', "
  //              "schedule : 'compact', "
  //              "chunk    : 10");

  // device.setup("mode        : 'OpenCL', "
  //              "platform_id : 0, "
  //              "device_id   : 1");

  // device.setup("mode      : 'CUDA', "
  //              "device_id : 0");
  //========================================================

  // Allocate memory on the device
  o_a  = device.malloc(entries, occa::dtype::float_);
  // Primitive types are available by template
  o_b  = device.malloc(entries, occa::dtype::get<float>());

  // We can also allocate memory without a dtype
  // WARNING: This will disable runtime type checking
  o_ab = device.malloc(entries * sizeof(float));

  // Compile the kernel at run-time
  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  // Copy memory to the device
  o_a.copyFrom(a);
  o_b.copyFrom(b);

  // Launch device kernel
  addVectors(entries, o_a, o_b, o_ab);

  // Copy result to the host
  o_ab.copyTo(ab);

  // Assert values
  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i])) {
      throw 1;
    }
  }

  // Free host memory
  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example adding two vectors"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"mode: 'Serial'\")")
      .withArg()
      .withDefaultValue("mode: 'OpenMP', threads: 8")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
