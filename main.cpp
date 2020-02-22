#include <iostream>
#include <vector>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

unsigned int intExp2(unsigned int p);

int main(int argc, const char **argv) {
    occa::json args = parseArgs(argc, argv);

    // These two should be input arguments
    const unsigned int N = 128; 
    const unsigned int levels = 4;
    const double tolerance = 1e-7;

    // Figuring out how many times we can coarsen
    unsigned int max_levels = 1;
    unsigned int N_at_h = 128;
    while (N_at_h%2 == 0 && N_at_h > 4){
        ++max_levels;
        N_at_h /= 2;
    }

    std::cout << "N: " << N << std::endl;
    std::cout << "Max levels: " << max_levels << std::endl;
    std::cout << "Levels: " << levels << std::endl;

    if (max_levels < levels) {
        std::cout << "Error: max levels is " << max_levels << " and requested levels is " << levels << ". Exiting." << std::endl;
        exit(-1);
    }

    // Vector containing all the grids
    std::vector<float> u(2*N, 1.0); // 1 is the initial guess

    // Vector containg number of nodes for all levels
    std::vector<float> N_h(max_levels, N);

    // Vector for keeping track of where all grids are in the big vector
    std::vector<float> offset(max_levels, 0.0);

    // Vector containing delta_x of each level
    std::vector<float> delta_x(max_levels, 1.0/(N - 1));

    for (unsigned int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1];
        delta_x[h] = 1.0/(N_h[h] - 1); // square here?
    }

    // Vector containing f at each point for the finest grid
    std::vector<float> f(N, 0.0);

    for (unsigned int i = 0; i < N; ++i) {
        f[i] = std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        u[offset[h]] = 0;           // u(0) = 0
        u[offset[h]+N_h[h]-1] = 0;  // u(1) = 0
    }

    // Jacobi iteration
    unsigned int h = 0;
    const double weight = 1.0; // With a weight of 1, the original Jacobi is recovered
    for (unsigned int n = 0; n < 1000; ++n) {
        for (unsigned int i = 1; i < N-1; ++i) {
            const unsigned int u_star = 0.5*(u[offset[h] + i + 1] + u[offset[h] + i - 1] + std::pow(delta_x[h], 2) * f[i]); // u_star could be made an array if we don't want to see u_i-1^(n+1), because as it is now u_i-1 gets updated before u_i
            u[offset[h] + i] += weight * (u_star - u[offset[h] + i]);
        }
    }



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

unsigned int intExp2(unsigned int p)
{
  if (p == 0) return 1;
  if (p == 1) return 2;

  unsigned int tmp = intExp2(p/2);
  if (p%2 == 0) return tmp * tmp;
  else return 2 * tmp * tmp;
}