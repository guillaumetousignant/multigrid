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

    // Figuring out how many times we can coarsen
    unsigned int max_levels = 1;
    unsigned int N_at_h = N;
    while (N_at_h%2 == 0 && N_at_h > 2){
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
    std::vector<float> u(2*N+max_levels-2, 1.0); // 1 is the initial guess

    // Vector containg number of elements for all levels
    std::vector<unsigned int> N_h(max_levels, N);

    // Vector for keeping track of where all grids are in the big vector
    std::vector<unsigned int> offset(max_levels, 0.0);

    // Vector containing delta_x of each level
    std::vector<float> delta_x(max_levels, 1.0/N);

    // Vector containing f at each point for the finest grid
    std::vector<float> f(2*N+max_levels-2, 0.0);

    // u_star is cached, so that it can be computed before the u_i start being updated
    std::vector<float> u_star(2*N+max_levels-2, 0.0);

    // Residuals are cached, so that it can run in parallel
    std::vector<float> r(2*N+max_levels-2, 0.0);

    for (unsigned int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1] + 1;
        delta_x[h] = 1.0/N_h[h];
    }

    for (unsigned int i = 0; i <= N; ++i) {
        f[offset[0] + i] = std::pow(delta_x[0], 2) * std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (unsigned int h = 0; h < max_levels; ++h) {
        u[offset[h]] = 0.0;           // u(0) = 0
        u[offset[h]+N_h[h]] = 0.0;    // u(1) = 0
        r[offset[h]] = 0.0;           // r(0) = 0
        r[offset[h]+N_h[h]] = 0.0;    // r(1) = 0
    }

    // Jacobi iteration
    const float weight = 2.0/(1.0 + std::sqrt(1.0 + std::pow(std::cos(M_PI * delta_x[0]), 2))); // With a weight of 1, the original Jacobi is recovered
    float residual = 1.0;
    unsigned int n = 0;
    unsigned int n_V = 0;
    const unsigned int n_relax_down = 100;    // Will actually do one more because of residuals calculation
    const unsigned int n_relax_up = 100;      // Will actually do one more because of residuals calculation
    unsigned int level = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    while (residual > tolerance) {
        ++n_V;

        // Relaxation steps
        for (unsigned int k = 0; k < n_relax_down; ++k){
            ++n;
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
            }
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                u[offset[level] + i] += weight * (u_star[offset[level] + i] - u[offset[level] + i]);
            }
        }

        // Calculate residuals
        ++n;
        for (unsigned int i = 1; i < N_h[level]; ++i) {
            u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
        }
        for (unsigned int i = 1; i < N_h[level]; ++i) {
            r[offset[level] + i] = weight * (u_star[offset[level] + i] - u[offset[level] + i]);
            u[offset[level] + i] += r[offset[level] + i];
        }

        // Going down
        while (level < levels-1){
            ++level;

            // Restriction
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                f[offset[level] + i] = 0.25*(r[offset[level-1] + 2*i - 1] + r[offset[level-1] + 2*i + 1] + 2.0*r[offset[level-1] + 2*i]);
                u[offset[level] + i] = 0.0; // Initial guess for the velocity correction
            }

            // Relaxation steps
            for (unsigned int k = 0; k < n_relax_down; ++k){
                ++n;
                for (unsigned int i = 1; i < N_h[level]; ++i) {
                    u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
                }
                for (unsigned int i = 1; i < N_h[level]; ++i) {
                    u[offset[level] + i] += weight * (u_star[offset[level] + i] - u[offset[level] + i]);
                }
            }

            // Calculate residuals
            ++n;
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
            }
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                r[offset[level] + i] = weight * (u_star[offset[level] + i] - u[offset[level] + i]);
                u[offset[level] + i] += r[offset[level] + i];
            }
        }

        // Solve fully here?

        // Going up
        while (level > 0){
            --level;
            
            // Prolongation
            for (unsigned int i = 0; i < N_h[level+1]; ++i) {
                u[offset[level] + 2*i] += u[offset[level+1] + i];
                u[offset[level] + 2*i + 1] += 0.5 * (u[offset[level+1] + i] + u[offset[level+1] + i + 1]);
            }

            // Relaxation steps
            for (unsigned int k = 0; k < n_relax_up; ++k){
                ++n;
                for (unsigned int i = 1; i < N_h[level]; ++i) {
                    u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
                }
                for (unsigned int i = 1; i < N_h[level]; ++i) {
                    u[offset[level] + i] += weight * (u_star[offset[level] + i] - u[offset[level] + i]);
                }
            }

            // Calculate residuals
            ++n;
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                u_star[offset[level] + i] = 0.5*(u[offset[level] + i + 1] + u[offset[level] + i - 1] + f[offset[level] + i]);
            }
            for (unsigned int i = 1; i < N_h[level]; ++i) {
                r[offset[level] + i] = weight * (u_star[offset[level] + i] - u[offset[level] + i]);
                u[offset[level] + i] += r[offset[level] + i];
            }
        }        

        // Norm
        residual = 0.0;
        for (unsigned int i = 1; i < N; ++i) {
            residual += std::pow(r[offset[level] + i], 2);
        }
        residual = std::sqrt(residual);

        // Max
        /*residual = 0.0;
        for (unsigned int i = 1; i < N; ++i) {
            residual = std::max(residual, std::abs(r[offset[level] + i]));
        }*/
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    // Display section
    double error = 0.0;
    for (unsigned int i = 1; i <= N; ++i) {
        error = std::max(error, std::abs(u[offset[0] + i] - std::sin(M_PI * i * delta_x[0])));
    }

    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (unsigned int i = 0; i <= N; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[0] + i] << " " << std::setw(15) << std::sin(M_PI * i * delta_x[0]) << " " << std::setw(15) << r[offset[0] + i] << " " << std::setw(15) << std::abs(u[offset[0] + i] - std::sin(M_PI * i * delta_x[0])) << std::endl;
    }

    std::cout << std::endl << "Iterations  residual norm  error norm    time taken [s]      steps" << std::endl;
    std::cout << n_V << " " << std::setw(15) << residual << " " << std::setw(15) << error << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 << " " << std::setw(15) << n << std::endl;



    /*
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
    */

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