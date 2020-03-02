#include <iostream>
#include <vector>
#include <chrono>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

int intExp2(int p);

void relaxation(std::vector<float> &u, std::vector<float> &u_star, const std::vector<float> &f, int N, int offset, double weight, int n_iter) {
    for (int k = 0; k < n_iter; ++k){
        for (int i = 1; i < N; ++i) {
            u_star[offset + i] = 0.5*(u[offset + i + 1] + u[offset + i - 1] + f[offset + i]);
        }
        for (int i = 1; i < N; ++i) {
            u[offset + i] += weight * (u_star[offset + i] - u[offset + i]);
        }
    }
}

void residuals(std::vector<float> &u, std::vector<float> &u_star, std::vector<float> &r, const std::vector<float> &f, int N, int offset, double weight) {
    for (int i = 1; i < N; ++i) {
        u_star[offset + i] = 0.5*(u[offset + i + 1] + u[offset + i - 1] + f[offset + i]);
    }
    for (int i = 1; i < N; ++i) {
        r[offset + i] = weight * (u_star[offset + i] - u[offset + i]);
        u[offset + i] += r[offset + i];
    }
}

void restriction(std::vector<float> &u, const std::vector<float> &r, std::vector<float> &f, int N, int offset_coarse, int offset_fine) {
    for (int i = 1; i < N; ++i) {
        f[offset_coarse + i] = 0.25*(r[offset_fine + 2*i - 1] + r[offset_fine + 2*i + 1] + 2.0*r[offset_fine + 2*i]);
        u[offset_coarse + i] = 0.0; // Initial guess for the velocity correction
    }
}

void prolongation(std::vector<float> &u, int N, int offset_coarse, int offset_fine) {
    for (int i = 0; i < N; ++i) {
        u[offset_fine + 2*i] += u[offset_coarse + i];
        u[offset_fine + 2*i + 1] += 0.5 * (u[offset_coarse + i] + u[offset_coarse + i + 1]);
    }
}

float residual_max(const std::vector<float> &r, int N, int offset) {
    float residual = 0.0;
    for (int i = 1; i < N; ++i) {
        residual = std::max(residual, std::abs(r[offset + i]));
    }
    return residual;
}

float residual_norm(const std::vector<float> &r, int N, int offset) {
    float residual = 0.0;
    for (int i = 1; i < N; ++i) {
            residual += std::pow(r[offset + i], 2);
        }
    return std::sqrt(residual);
}

float analytical(float x) {
    return std::sin(M_PI * x);
}

float error(const std::vector<float> &u, int offset, float delta_x, int i) {
    return u[offset + i] - analytical(i * delta_x);
}

float error_max(const std::vector<float> &u, int N, int offset, float delta_x) {
    float value = 0.0;
    for (int i = 1; i <= N; ++i) {
        value = std::max(value, std::abs(u[offset + i] - analytical(i * delta_x)));
    }
    return value;
}

float error_norm(const std::vector<float> &u, int N, int offset, float delta_x) {
    float value = 0.0;
    for (int i = 1; i <= N; ++i) {
        value += std::pow(error(u, offset, delta_x, i), 2);
    }
    return std::sqrt(value);
}

int main(int argc, const char **argv) {
    // Parameters
    int N = 16;        // First command line input
    int levels = 4;    // Second command line input
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
    occa::kernel initialFConditions_GPU, initialConditions_GPU;
    occa::kernel relaxation_GPU, residuals_GPU;
    occa::kernel restriction_GPU, prolongation_GPU;
    occa::kernel reduction_max_GPU, reduction_norm_GPU;
    occa::memory f_GPU, u_GPU, u_star_GPU, r_GPU;
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

    // Reduction parameters
    unsigned int block   = 256; // Block size ALSO CHANGE IN KERNEL
    unsigned int max_blocks  = (N + block - 1)/block;    
    std::vector<float> block_sum(max_blocks, 0.0);
    block_sum_GPU = device.malloc(max_blocks, occa::dtype::float_);

    // Figuring out how many times we can coarsen
    int max_levels = 1;
    int N_at_h = N;
    while (N_at_h%2 == 0 && N_at_h > 2){
        ++max_levels;
        N_at_h /= 2;
    }

    std::cout << "N: " << N << std::endl;
    std::cout << "Max levels: " << max_levels << std::endl;
    std::cout << "Levels: " << levels << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;

    if (max_levels < levels) {
        std::cout << "Error: max levels is " << max_levels << " and requested levels is " << levels << ". Exiting." << std::endl;
        exit(-1);
    }

    // Initial values
    float u_0 = 0.0; // u(0) = 0
    float u_i = 1.0; // 1 is the initial guess
    float u_N = 0.0; // u(1) = 0
    float r_0 = 0.0; // r(0) = 0
    float r_i = 1.0; // 1 is the intial residual, so the while loop is entered
    float r_N = 0.0; // r(1) = 0

    // Vector of u containing all the grids
    std::vector<float> u(2*N+max_levels-2, u_i); 

    // Vector containg number of elements for all levels
    std::vector<int> N_h(max_levels, N);

    // Vector for keeping track of where all grids are in the big vector
    std::vector<int> offset(max_levels, 0.0);

    // Vector containing delta_x of each level
    std::vector<float> delta_x(max_levels, 1.0/N);

    // Vector containing f at each point for the finest grid
    std::vector<float> f(2*N+max_levels-2, 0.0);

    // u_star is cached, so that it can be computed before the u_i start being updated
    std::vector<float> u_star(2*N+max_levels-2, 0.0);

    // Residuals are cached, so that it can run in parallel
    std::vector<float> r(2*N+max_levels-2, r_i);

    for (int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1] + 1;
        delta_x[h] = 1.0/N_h[h];
    }

    // f initial conditions
    for (int i = 0; i <= N_h[0]; ++i) {
        f[offset[0] + i] = std::pow(delta_x[0], 2) * std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (int h = 0; h < max_levels; ++h) {
        u[offset[h]] = u_0;           
        u[offset[h]+N_h[h]] = u_N;    
        r[offset[h]] = r_0;           
        r[offset[h]+N_h[h]] = r_N;    
    }

    // GPU vectors init
    f_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);
    u_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);
    u_star_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);
    r_GPU = device.malloc(2*N+max_levels-2, occa::dtype::float_);

    // Compile the kernel at run-time
    initialFConditions_GPU = device.buildKernel("multigrid.okl",
                                    "initialFConditions");
    initialConditions_GPU = device.buildKernel("multigrid.okl",
                                    "initialConditions");
    relaxation_GPU = device.buildKernel("multigrid.okl",
                                    "relaxation");
    residuals_GPU = device.buildKernel("multigrid.okl",
                                    "residuals");
    restriction_GPU = device.buildKernel("multigrid.okl",
                                    "restriction");
    prolongation_GPU = device.buildKernel("multigrid.okl",
                                    "prolongation");
    reduction_max_GPU = device.buildKernel("multigrid.okl",
                                    "reduction_max");
    reduction_norm_GPU = device.buildKernel("multigrid.okl",
                                    "reduction_norm");

    // Initial f conditions
    initialFConditions_GPU(N_h[0], delta_x[0], f_GPU);

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        initialConditions_GPU(N_h[h], offset[h], u_i, u_0, u_N, u_GPU, r_GPU); // Can't send doubles, they won't be cast.
    }

    // Jacobi iteration
    const float weight = 2.0/(1.0 + std::sqrt(1.0 + std::pow(std::cos(M_PI * delta_x[0]), 2))); // With a weight of 1, the original Jacobi is recovered
    int n = 0;
    int n_V = 0;
    const int n_relax_down = 5;    // Will actually do one more because of residuals calculation
    const int n_relax_up = 5;      // Will actually do one more because of residuals calculation
    int n_GPU = 0;
    int n_V_GPU = 0;
    double residual_GPU = 1000000.0;

    // GPU part
    auto t_start_GPU = std::chrono::high_resolution_clock::now();
    while (residual_GPU > tolerance) {
        ++n_V_GPU;

        // Relaxation steps
        for (int i = 0; i < n_relax_down; ++i){
            relaxation_GPU(N_h[0], offset[0], weight, f_GPU, u_GPU, u_star_GPU);
        }
        n_GPU += n_relax_down;

        // Calculate residuals
        residuals_GPU(N_h[0], offset[0], weight, f_GPU, u_GPU, u_star_GPU, r_GPU);
        ++n_GPU;

        // Going down
        for (int level = 1; level < levels; ++level) {

            // Restriction
            restriction_GPU(N_h[level], offset[level], offset[level - 1], f_GPU, u_GPU, r_GPU);

            // Relaxation steps
            for (int i = 0; i < n_relax_down; ++i){
                relaxation_GPU(N_h[level], offset[level], weight, f_GPU, u_GPU, u_star_GPU);
            }
            n_GPU += n_relax_down;

            // Calculate residuals
            residuals_GPU(N_h[level], offset[level], weight, f_GPU, u_GPU, u_star_GPU, r_GPU);
            ++n_GPU;
        }

        // Solve fully here?

        // Going up
        for (int level = levels - 2; level >= 0; --level){
            
            // Prolongation
            prolongation_GPU(N_h[level + 1], offset[level + 1], offset[level], u_GPU);

            // Relaxation steps
            for (int i = 0; i < n_relax_up; ++i){
                relaxation_GPU(N_h[level], offset[level], weight, f_GPU, u_GPU, u_star_GPU);
            }
            n_GPU += n_relax_up;

            // Calculate residuals
            residuals_GPU(N_h[level], offset[level], weight, f_GPU, u_GPU, u_star_GPU, r_GPU);
            ++n_GPU;
        }       

        reduction_norm_GPU(N_h[0], offset[0], r_GPU, block_sum_GPU);
        // Host <- Device
        block_sum_GPU.copyTo(block_sum.data());

        // Finalize the reduction in the host
        residual_GPU = 0.0;
        for (unsigned int i = 0; i < (N + block - 1)/block; ++i) {
            residual_GPU += block_sum[i];
        }
        residual_GPU = std::sqrt(residual_GPU);
    }
    auto t_end_GPU = std::chrono::high_resolution_clock::now();

    // CPU part
    auto t_start = std::chrono::high_resolution_clock::now();
    while (residual_norm(r, N_h[0], offset[0]) > tolerance) {
        ++n_V;

        // Relaxation steps
        relaxation(u, u_star, f, N_h[0], offset[0], weight, n_relax_down);
        n += n_relax_down;

        // Calculate residuals
        residuals(u, u_star, r, f, N_h[0], offset[0], weight);
        ++n;

        // Going down
        for (int level = 1; level < levels; ++level) {

            // Restriction
            restriction(u, r, f, N_h[level], offset[level], offset[level - 1]);

            // Relaxation steps
            relaxation(u, u_star, f, N_h[level], offset[level], weight, n_relax_down);
            n += n_relax_down;

            // Calculate residuals
            residuals(u, u_star, r, f, N_h[level], offset[level], weight);
            ++n;
        }

        // Solve fully here?

        // Going up
        for (int level = levels - 2; level >= 0; --level){
            
            // Prolongation
            prolongation(u, N_h[level+1], offset[level + 1], offset[level]);

            // Relaxation steps
            relaxation(u, u_star, f, N_h[level], offset[level], weight, n_relax_up);
            n += n_relax_up;

            // Calculate residuals
            residuals(u, u_star, r, f, N_h[level], offset[level], weight);
            ++n;
        }        
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    // Display section
    std::cout << std::endl << "CPU result" << std::endl;
    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (int i = 0; i <= N_h[0]; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[0] + i] << " " << std::setw(15) << analytical(i * delta_x[0]) << " " << std::setw(15) << r[offset[0] + i] << " " << std::setw(15) << error(u, offset[0], delta_x[0], i) << std::endl;
    }

    // Host <- Device
    std::vector<float> u_GPU_local(u.size());
    std::vector<float> r_GPU_local(r.size());
    u_GPU.copyTo(u_GPU_local.data());
    r_GPU.copyTo(r_GPU_local.data());

    std::cout << std::endl << "GPU result" << std::endl;
    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (unsigned int i = 0; i <= N_h[0]; ++i) {
        std::cout << i << " " << std::setw(15) << u_GPU_local[offset[0] + i] << " " << std::setw(15) << analytical(i * delta_x[0]) << " " << std::setw(15) << r_GPU_local[offset[0] + i] << " " << std::setw(15) << error(u_GPU_local, offset[0], delta_x[0], i) << std::endl;
    }

    std::cout << std::endl << "CPU result" << std::endl;
    std::cout << "Iterations  residual norm  error norm    time taken [s]      steps" << std::endl;
    std::cout << n_V << " " << std::setw(15) << residual_norm(r, N_h[0], offset[0]) << " " << std::setw(15) << error_norm(u, N_h[0], offset[0], delta_x[0]) << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 << " " << std::setw(15) << n << std::endl;

    std::cout << std::endl << "GPU result" << std::endl;
    std::cout << "Iterations  residual norm  error norm    time taken [s]      steps" << std::endl;
    std::cout << n_V_GPU << " " << std::setw(15) << residual_norm(r_GPU_local, N_h[0], offset[0]) << " " << std::setw(15) << error_norm(u_GPU_local, N_h[0], offset[0], delta_x[0]) << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end_GPU-t_start_GPU).count()/1000.0 << " " << std::setw(15) << n_GPU << std::endl;

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

int intExp2(int p) {
  if (p == 0) return 1;
  if (p == 1) return 2;

  int tmp = intExp2(p/2);
  if (p%2 == 0) return tmp * tmp;
  else return 2 * tmp * tmp;
}