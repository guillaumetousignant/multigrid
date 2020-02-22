#include <iostream>
#include <vector>
#include <chrono>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

unsigned int intExp2(unsigned int p);

int main(int argc, const char **argv) {
    occa::json args = parseArgs(argc, argv);

    // These two should be input arguments
    const unsigned int N = 256; 
    const unsigned int levels = 4;
    const float tolerance = 1e-7;

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

    // Vector of u containing all the grids
    std::vector<float> u(2*N, 1.0); // 1 is the initial guess

    // Vector containg number of nodes for all levels
    std::vector<float> N_h(max_levels, N);

    // Vector for keeping track of where all grids are in the big vector
    std::vector<float> offset(max_levels, 0.0);

    // Vector containing delta_x of each level
    std::vector<float> delta_x(max_levels, 1.0/(N - 1));

    // Vector containing f at each point for the finest grid
    std::vector<float> f(N, 0.0);

    // u_star is cached, so that it can be computed before the u_i start being updated
    std::vector<float> u_star(2*N, 0.0);

    // Residuals are cached, so that it can run in parallel
    std::vector<float> r(2*N, 0.0);

    for (unsigned int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1];
        delta_x[h] = 1.0/(N_h[h] - 1); // square here?
    }

    for (unsigned int i = 0; i < N; ++i) {
        f[i] = std::pow(delta_x[0], 2) * std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        u[offset[h]] = 0;           // u(0) = 0
        u[offset[h]+N_h[h]-1] = 0;  // u(1) = 0
    }

    // Jacobi iteration
    unsigned int h = 0;
    const float weight = 1.0; // With a weight of 1, the original Jacobi is recovered
    float residual = 1.0;
    unsigned int n = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    while (residual > tolerance) {
        ++n;
        for (unsigned int i = 1; i < N-1; ++i) {
            u_star[offset[h] + i] = 0.5*(u[offset[h] + i + 1] + u[offset[h] + i - 1] + f[i]);
        }
        for (unsigned int i = 1; i < N-1; ++i) {
            r[offset[h] + i] = weight * (u_star[offset[h] + i] - u[offset[h] + i]);
            u[offset[h] + i] += r[offset[h] + i];
        }

        // Norm
        /*residual = 0.0;
        for (unsigned int i = 1; i < N-1; ++i) {
            residual += std::pow(r[offset[h] + i], 2);
        }
        residual = std::sqrt(residual);*/

        // Max
        residual = 0.0;
        for (unsigned int i = 1; i < N-1; ++i) {
            residual = std::max(residual, std::abs(r[offset[h] + i]));
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    // Display section
    double error = 0.0;
    for (unsigned int i = 1; i < N-1; ++i) {
        error = std::max(error, std::abs(u[offset[0] + i] - std::sin(M_PI * i * delta_x[h])));
    }

    std::cout << "i    numerical    analytical    residual    error" << std::endl;
    for (unsigned int i = 0; i < N; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[h] + i] << " " << std::setw(15) << r[offset[h] + i] << " " << std::setw(15) << std::sin(M_PI * i * delta_x[h]) << " " << std::setw(15) << std::abs(u[offset[h] + i] - std::sin(M_PI * i * delta_x[h])) << std::endl;
    }

    std::cout << std::endl << "Iterations    max residual    max error    time taken [s]" << std::endl;
    std::cout << n << " " << std::setw(15) << residual << " " << std::setw(15) << error << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 << std::endl;




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
    /*for (int i = 0; i < 5; ++i) {
        std::cout << i << ": " << ab[i] << '\n';
    }*/
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