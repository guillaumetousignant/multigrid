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

    if (max_levels < levels) {
        std::cout << "Error: max levels is " << max_levels << " and requested levels is " << levels << ". Exiting." << std::endl;
        exit(-1);
    }

    // Vector of u containing all the grids
    std::vector<float> u(2*N+max_levels-2, 1.0); // 1 is the initial guess

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
    std::vector<float> r(2*N+max_levels-2, 1.0);

    for (int h = 1; h < max_levels; ++h) {
        N_h[h] /= intExp2(h);
        offset[h] = offset[h-1] + N_h[h-1] + 1;
        delta_x[h] = 1.0/N_h[h];
    }

    for (int i = 0; i <= N; ++i) {
        f[offset[0] + i] = std::pow(delta_x[0], 2) * std::pow(M_PI, 2) * std::sin(M_PI * i * delta_x[0]);
    }

    // Initial conditions
    for (int h = 0; h < max_levels; ++h) {
        u[offset[h]] = 0.0;           // u(0) = 0
        u[offset[h]+N_h[h]] = 0.0;    // u(1) = 0
        r[offset[h]] = 0.0;           // r(0) = 0
        r[offset[h]+N_h[h]] = 0.0;    // r(1) = 0
    }

    // Jacobi iteration
    const float weight = 2.0/(1.0 + std::sqrt(1.0 + std::pow(std::cos(M_PI * delta_x[0]), 2))); // With a weight of 1, the original Jacobi is recovered
    int n = 0;
    int n_V = 0;
    const int n_relax_down = 5;    // Will actually do one more because of residuals calculation
    const int n_relax_up = 5;      // Will actually do one more because of residuals calculation

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
    std::cout << "i      numerical      analytical        residual           error" << std::endl;
    for (int i = 0; i <= N; ++i) {
        std::cout << i << " " << std::setw(15) << u[offset[0] + i] << " " << std::setw(15) << analytical(i * delta_x[0]) << " " << std::setw(15) << r[offset[0] + i] << " " << std::setw(15) << error(u, offset[0], delta_x[0], i) << std::endl;
    }

    std::cout << std::endl << "Iterations  residual norm  error norm    time taken [s]      steps" << std::endl;
    std::cout << n_V << " " << std::setw(15) << residual_norm(r, N_h[0], offset[0]) << " " << std::setw(15) << error_norm(u, N_h[0], offset[0], delta_x[0]) << " " << std::setw(15) << std::chrono::duration<double, std::milli>(t_end-t_start).count()/1000.0 << " " << std::setw(15) << n << std::endl;

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