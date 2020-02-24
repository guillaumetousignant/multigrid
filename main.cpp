#include <iostream>
#include <vector>
#include <chrono>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

unsigned int intExp2(unsigned int p);

int main(int argc, const char **argv) {
    // Parameters
    unsigned int N = 16;        // First command line input
    unsigned int levels = 4;    // Second command line input
    float tolerance = 1e-10;    // Third command line input
    int argc_occa = argc;
    const char **argv_occa = argv;

    if (argc > 3) {
        N = std::stoi(argv[1]);
        levels = std::stoi(argv[2]);
        tolerance = std::stod(argv[3]); // Using stod and casting to float cause I'll forget to change it if I change to doubles
        argc_occa -= 3;
        argv_occa = argv + 3;
    }
    else if (argc > 2) {
        N = std::stoi(argv[1]);
        levels = std::stoi(argv[2]);
        argc_occa -= 2;
        argv_occa = argv + 2;
    }
    else if (argc > 1) {
        N = std::stoi(argv[1]);
        argc_occa -= 1;
        argv_occa = argv + 1;
    }
    occa::json args = parseArgs(argc_occa, argv_occa);

    occa::device device;
    occa::kernel initialFConditions, initialConditions;
    occa::kernel weighedJacobiTop;
    occa::kernel reduction;
    occa::memory f_GPU, u_GPU, u_star_GPU, r_GPU;
    occa::memory N_h_GPU, offset_GPU, delta_x_GPU;
    occa::memory block_sum_GPU;

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

    // Figuring out how many times we can coarsen
    unsigned int max_levels = 1;
    unsigned int N_at_h = N;
    while (N_at_h%2 == 0 && N_at_h > 2){
        ++max_levels;
        N_at_h /= 2;
    }

    // Reduction parameters
    unsigned int block   = 256; // Block size ALSO CHANGE IN KERNEL
    unsigned int max_blocks  = (N + block - 1)/block;    
    std::vector<float> block_sum(max_blocks, 0.0);
    block_sum_GPU = device.malloc(max_blocks, occa::dtype::float_);

    std::cout << "N: " << N << std::endl;
    std::cout << "Max levels: " << max_levels << std::endl;
    std::cout << "Levels: " << levels << std::endl;

    if (max_levels < levels) {
        std::cout << "Error: max levels is " << max_levels << " and requested levels is " << levels << ". Exiting." << std::endl;
        exit(-1);
    }

    // Vector of u containing all the grids
    std::vector<float> u(2*N+max_levels-2, 1.0); // 1 is the initial guess

    // Vector containg number of elements for all levels
    std::vector<unsigned int> N_h(max_levels, N);

    // Vector for keeping track of where all grids are in the big vector
    std::vector<unsigned int> offset(max_levels, 0.0);

    // Vector containing delta_x of each level
    std::vector<float> delta_x(max_levels, 1.0/N);

    // Vector containing f at each point for the finest grid
    std::vector<float> f(N+1, 0.0);

    // u_star is cached, so that it can be computed before the u_i start being updated
    std::vector<float> u_star(2*N+max_levels-2, 0.0);

    // Residuals are cached, so that it can run in parallel
    std::vector<float> r(2*N+max_levels-2, 0.0);

    // Initial grid parameters
    for (unsigned int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1] + 1;
        delta_x[h] = 1.0/N_h[h];
    }

    // Initial f conditions
    for (unsigned int i = 0; i <= N; ++i) {
        f[i] = std::pow(delta_x[0], 2) * std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        u[offset[h]] = 0.0;           // u(0) = 0
        u[offset[h]+N_h[h]] = 0.0;    // u(1) = 0
        r[offset[h]] = 0.0;           // r(0) = 0
        r[offset[h]+N_h[h]] = 0.0;    // r(1) = 0
    }

    // GPU vectors init
    N_h_GPU = device.malloc(max_levels, occa::dtype::uint32);
    offset_GPU = device.malloc(max_levels, occa::dtype::uint32);
    delta_x_GPU = device.malloc(max_levels, occa::dtype::float_);
    N_h_GPU.copyFrom(N_h.data());
    offset_GPU.copyFrom(offset.data());
    delta_x_GPU.copyFrom(delta_x.data());

    f_GPU = device.malloc(N+1, occa::dtype::float_);
    u_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);
    u_star_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);
    r_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);

    // Compile the kernel at run-time
    initialFConditions = device.buildKernel("jacobi.okl",
                                    "initialFConditions");
    initialConditions = device.buildKernel("jacobi.okl",
                                    "initialConditions");
    weighedJacobiTop = device.buildKernel("jacobi.okl",
                                    "weighedJacobiTop");
    reduction = device.buildKernel("jacobi.okl",
                                    "reduction");

    // Initial f conditions
    initialFConditions(N, delta_x[0], f_GPU);

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        initialConditions(N_h[h], offset[h], 1.0, 0.0, 0.0, u_GPU, r_GPU);
    }

    // Jacobi iteration
    unsigned int h = 0;
    const float weight = 2.0/(1.0 + std::sqrt(1.0 + std::pow(std::cos(M_PI * delta_x[0]), 2))); // With a weight of 1, the original Jacobi is recovered
    float residual = 1.0;
    float residual_GPU = 1.0;
    unsigned int n = 0;
    unsigned int n_GPU = 0;

    auto t_start_GPU = std::chrono::high_resolution_clock::now();
    while (residual_GPU > tolerance) {
        ++n_GPU;
        weighedJacobiTop(N_h[h], offset[h], weight, f_GPU, u_GPU, u_star_GPU, r_GPU);
        reduction(N_h[h], offset[h], r_GPU, block_sum_GPU);

        // Host <- Device
        block_sum_GPU.copyTo(block_sum.data());

        // Finalize the reduction in the host
        residual_GPU = 0.0;
        for (unsigned int i = 0; i < (N + block - 1)/block; ++i) {
            residual_GPU = std::max(residual_GPU, block_sum[i]);
        }
    }
    auto t_end_GPU = std::chrono::high_resolution_clock::now();

    auto t_start = std::chrono::high_resolution_clock::now();
    while (residual > tolerance) {
        ++n;
        for (unsigned int i = 1; i < N; ++i) {
            u_star[offset[h] + i] = 0.5*(u[offset[h] + i + 1] + u[offset[h] + i - 1] + f[i]);
        }
        for (unsigned int i = 1; i < N; ++i) {
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
        for (unsigned int i = 1; i < N; ++i) {
            residual = std::max(residual, std::abs(r[offset[h] + i]));
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    // Display section
    double error = 0.0;
    for (unsigned int i = 1; i <= N; ++i) {
        error = std::max(error, std::abs(u[offset[0] + i] - std::sin(M_PI * i * delta_x[h])));
    }

    std::cout << std::endl << "CPU result" << std::endl;
    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (unsigned int i = 0; i <= N; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[h] + i] << " " << std::setw(15) << std::sin(M_PI * i * delta_x[h]) << std::setw(15) << r[offset[h] + i] << " " << " " << std::setw(15) << std::abs(u[offset[h] + i] - std::sin(M_PI * i * delta_x[h])) << std::endl;
    }

    std::cout << std::endl << "Iterations  max residual   max error       time taken [s]" << std::endl;
    std::cout << n << " " << std::setw(15) << residual << " " << std::setw(15) << error << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 << std::endl;

    // Host <- Device
    u_GPU.copyTo(u.data());
    r_GPU.copyTo(r.data());

    double error_GPU = 0.0;
    for (unsigned int i = 1; i <= N; ++i) {
        error_GPU = std::max(error_GPU, std::abs(u[offset[0] + i] - std::sin(M_PI * i * delta_x[h])));
    }

    std::cout << std::endl << "GPU result" << std::endl;
    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (unsigned int i = 0; i <= N; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[h] + i] << " " << std::setw(15) << std::sin(M_PI * i * delta_x[h]) << std::setw(15) << r[offset[h] + i] << " " << " " << std::setw(15) << std::abs(u[offset[h] + i] - std::sin(M_PI * i * delta_x[h])) << std::endl;
    }

    std::cout << std::endl << "Iterations  max residual   max error       time taken [s]" << std::endl;
    std::cout << n_GPU << " " << std::setw(15) << residual_GPU << " " << std::setw(15) << error_GPU << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end_GPU-t_start_GPU).count()/1000.0 << std::endl;


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
                            "Device properties (default: \"mode: 'OpenMP', threads: 8\")")
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