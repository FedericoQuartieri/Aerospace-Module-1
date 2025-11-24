#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


// Include C headers in extern "C" block for C++ compatibility
extern "C"
{
#include "g_field.h"
#include "velocity_field.h"
#include "pressure.h"
#include "solve.h"
#include "forcing_parser.h"
#include <math.h>
}

/* ----- Parameters (tune as you wish) ----- */

static const double Re       = 100.0;  /* Reynolds number */
static const double k_over_nu = 0.0;   /* k/nu term in the PDE, set if needed */

/* ----- Forcing function compatible with forcing_function_t ----- */
/* forcing_function_t: double (*)(double x, double y, double z, double t, int component) */

static double manufactured_forcing(double x, double y, double z, double t, int component)
{
    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t), ct = cos(t);

    /* Exact solution components */
    double ux = st * sx * sy * sz;
    double uy = st * cx * cy * cz;
    double uz = st * cx * sy * (cz + sz);

    /* Time derivative du/dt */
    double dudt_x = ct * sx * sy * sz;
    double dudt_y = ct * cx * cy * cz;
    double dudt_z = ct * cx * sy * (cz + sz);

    /* Pressure gradient components */
    double factor = -3.0 / Re;
    double gp_x = factor * st * (-sx * sy * (sz - cz));
    double gp_y = factor * st * (cx * cy * (sz - cz));
    double gp_z = factor * st * (cx * sy * (cz + sz));

    /* Laplacian: ∇²u = -3u for this manufactured solution */
    double coeff_u = 3.0 / Re + k_over_nu;

    /* f = du/dt + (3/Re) u + (k/nu) u + ∇p */
    switch(component) {
        case 0: // f_x
            return dudt_x + coeff_u * ux + gp_x;
        case 1: // f_y
            return dudt_y + coeff_u * uy + gp_y;
        case 2: // f_z
            return dudt_z + coeff_u * uz + gp_z;
        default:
            return 0.0;
    }
}

/* ----- Exact solution for verification ----- */

static double u_exact_component(double x, double y, double z, double t, int component)
{
    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    switch(component) {
        case 0: // u_x
            return st * sx * sy * sz;
        case 1: // u_y
            return st * cx * cy * cz;
        case 2: // u_z
            return st * cx * sy * (cz + sz);
        default:
            return 0.0;
    }
}

static double p_exact_value(double x, double y, double z, double t)
{
    double cx = cos(x);
    double sy = sin(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    double factor = -3.0 / Re;
    return factor * st * cx * sy * (sz - cz);
}


/* ----- Helper function to read VTK file ----- */ 
bool read_last_vtk_file(VelocityField &U_numerical, Pressure &P_numerical)
{
    int last_step = STEPS - 1;

    std::stringstream filename;
    filename << "../build/output/solution_0000" << last_step << ".vti";

    std::ifstream file(filename.str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename.str() << std::endl;
        return false;
    }

    std::string line;

    // 1. Find <AppendedData>
    while (std::getline(file, line)) {
        if (line.find("<AppendedData") != std::string::npos) {
            std::cout << "[DEBUG] Found <AppendedData>" << std::endl;
            break;
        }
    }

    // 2. Skip whitespace until we find '_'
    char ch;
    bool found_underscore = false;
    while (file.get(ch)) {
        if (ch == '_') {
            found_underscore = true;
            std::cout << "[DEBUG] Found '_' marker" << std::endl;
            break;
        }
        // Skip whitespace (spaces, newlines, etc.)
        if (!std::isspace(ch)) {
            std::cerr << "[ERROR] Expected whitespace or '_', got '" << ch << "' (ASCII " << (int)ch << ")\n";
            return false;
        }
    }

    if (!found_underscore) {
        std::cerr << "[ERROR] Could not find '_' marker after <AppendedData>\n";
        return false;
    }

    std::streampos data_start = file.tellg();
    std::cout << "[DEBUG] Binary data starts at position: " << data_start << std::endl;

    //file.seekg(data_start);

    uint32_t block_size = 0;
    const uint32_t expected = WIDTH * HEIGHT * DEPTH * sizeof(DTYPE);

    auto read_block = [&](const char* label, char* dst)
    {
        std::cout << "\n[DEBUG] Reading block: " << label << std::endl;

        file.read(reinterpret_cast<char*>(&block_size), sizeof(uint32_t));

        std::cout << "[DEBUG]   block_size read = " << block_size << std::endl;
        std::cout << "[DEBUG]   expected       = " << expected << std::endl;

        if (!file.good()) {
            std::cout << "[ERROR] file.read(block_size) failed while reading header of " << label << std::endl;
            return false;
        }

        if (block_size != expected) {
            std::cout << "[ERROR] Block size mismatch for " << label
                      << ". block_size=" << block_size
                      << " expected=" << expected << std::endl;
            return false;
        }

        // read block_size bytes
        file.read(dst, block_size);

        std::cout << "[DEBUG]   bytes actually read = " << file.gcount()
                  << " (should be " << block_size << ")" << std::endl;

        if (!file.good()) {
            std::cout << "[ERROR] file.read(data) failed while reading data for " << label << std::endl;
            return false;
        }

        return true;
    };

    // ---- PRESSURE ----
    if (!read_block("Pressure", (char*)P_numerical.p)) return false;

    // ---- U_x ----
    if (!read_block("Velocity_x", (char*)U_numerical.v_x)) return false;

    // ---- U_y ----
    if (!read_block("Velocity_y", (char*)U_numerical.v_y)) return false;

    // ---- U_z ----
    if (!read_block("Velocity_z", (char*)U_numerical.v_z)) return false;

    std::cout << "\n[DEBUG] Successfully read all VTI fields\n";

    return true;
} 


TEST(ManufacturedSolution, VelocitySystemConvergenceFromFile)
{
    // Initialize pressure
    Pressure pressure;
    initialize_pressure(&pressure);

    // Set exact pressure at t=0
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                pressure.p[idx] = p_exact_value(x, y, z, 0.0);
            }
        }
    }

    // Initialize the 3 velocity fields
    VelocityField Eta, Zeta, U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

    // Set exact initial velocity at t=0
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                
                U.v_x[idx] = u_exact_component(x, y, z, 0.0, 0);
                U.v_y[idx] = u_exact_component(x, y, z, 0.0, 1);
                U.v_z[idx] = u_exact_component(x, y, z, 0.0, 2);
                
                Eta.v_x[idx] = U.v_x[idx];
                Eta.v_y[idx] = U.v_y[idx];
                Eta.v_z[idx] = U.v_z[idx];
                
                Zeta.v_x[idx] = U.v_x[idx];
                Zeta.v_y[idx] = U.v_y[idx];
                Zeta.v_z[idx] = U.v_z[idx];
            }
        }
    }

    // Set K (permeability)
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE);
    for (size_t i = 0; i < GRID_ELEMENTS; i++) {
        K[i] = 1.0; // Uniform permeability
    }

    // Compute Beta and Gamma
    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE);
    for (size_t idx = 0; idx < GRID_ELEMENTS; idx++) {
        Beta[idx] = 1.0 + (DT * NU) / (2.0 * K[idx]);
        Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
    }

    // Initialize G field
    GField g_field;
    initialize_g_field(&g_field);

    // Boundary conditions (can be zero or exact solution at boundaries)
    DTYPE *u_BC_current = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));
    DTYPE *u_BC_second = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));
    DTYPE *u_BC_third = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));

    // Run solver with manufactured forcing - write every timestep for accuracy
    int write_frequency = 1;
    bool full_output = false;
    solve(g_field, manufactured_forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, 
          u_BC_current, u_BC_second, u_BC_third, write_frequency, full_output, nullptr, nullptr);

    // Read numerical solution from last VTK file
    VelocityField U_numerical;
    Pressure P_numerical;
    initialize_velocity_field(&U_numerical);
    initialize_pressure(&P_numerical);
    
    ASSERT_TRUE(read_last_vtk_file(U_numerical, P_numerical)) << "Failed to read VTK output file";

    // Verify solution at final time
    double t_final = STEPS * DT;
    
    // Error metrics
    double max_error_abs = 0.0;      // L∞ norm (maximum absolute error)
    double sum_error_sq = 0.0;       // For L2 norm
    double sum_exact_sq = 0.0;       // For relative L2 norm
    int n_interior_points = 0;
    
    std::cout << "\n[DEBUG] Computing errors at t_final = " << t_final << std::endl;
    std::cout << "[DEBUG] Reading from VTI file" << std::endl;
    
    for (int k = 1; k < DEPTH-1; k++) {
        for (int j = 1; j < HEIGHT-1; j++) {
            for (int i = 1; i < WIDTH-1; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                
                // Exact solution at final time
                double ux_exact = u_exact_component(x, y, z, t_final, 0);
                double uy_exact = u_exact_component(x, y, z, t_final, 1);
                double uz_exact = u_exact_component(x, y, z, t_final, 2);
                
                // Numerical solution from VTI file
                double ux_num = U_numerical.v_x[idx];
                double uy_num = U_numerical.v_y[idx];
                double uz_num = U_numerical.v_z[idx];
                
                // Absolute errors per component
                double error_x = fabs(ux_num - ux_exact);
                double error_y = fabs(uy_num - uy_exact);
                double error_z = fabs(uz_num - uz_exact);
                
                // L∞ norm: maximum component-wise error
                double point_max_error = fmax(error_x, fmax(error_y, error_z));
                max_error_abs = fmax(max_error_abs, point_max_error);
                
                // Accumulate for L2 norm
                sum_error_sq += error_x * error_x + error_y * error_y + error_z * error_z;
                sum_exact_sq += ux_exact * ux_exact + uy_exact * uy_exact + uz_exact * uz_exact;
                
                n_interior_points++;
            }
        }
    }
    
    // Compute L2 norm: sqrt(sum(error²) / N)
    double error_L2 = sqrt(sum_error_sq / n_interior_points);
    
    // Compute magnitude of exact solution: sqrt(sum(exact²) / N)
    double exact_L2 = sqrt(sum_exact_sq / n_interior_points);
    
    // Relative L2 error: ||error||_L2 / ||exact||_L2
    double error_L2_relative = error_L2 / exact_L2;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Manufactured Solution Error Analysis  " << std::endl;
    std::cout << "  (From File - VTI output)              " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Time:                    t = " << std::fixed << std::setprecision(6) << t_final << std::endl;
    std::cout << "Interior points:         N = " << n_interior_points << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "L∞ norm (max abs error): " << std::scientific << std::setprecision(6) << max_error_abs << std::endl;
    std::cout << "L2 norm (RMS error):     " << std::scientific << std::setprecision(6) << error_L2 << std::endl;
    std::cout << "L2 exact (RMS solution): " << std::scientific << std::setprecision(6) << exact_L2 << std::endl;
    std::cout << "Relative L2 error:       " << std::scientific << std::setprecision(6) 
              << error_L2_relative << " (" << std::fixed << std::setprecision(2) 
              << error_L2_relative * 100.0 << "%)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Expect convergence - use both L∞ and relative L2 error
    EXPECT_LT(error_L2_relative, 0.1) 
        << "Relative L2 error should be < 10%";
    EXPECT_LT(max_error_abs, 1.0) 
        << "Maximum absolute error (L∞) should be reasonable";

    // Cleanup
    free(K);
    free(Beta);
    free(Gamma);
    free(u_BC_current);
    free(u_BC_second);
    free(u_BC_third);
    free_pressure(&pressure);
    free_pressure(&P_numerical);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_velocity_field(&U_numerical);
    free_g_field(&g_field);
}


TEST(ManufacturedSolution, VelocitySystemConvergenceFromMemory)
{
    // Initialize pressure
    Pressure pressure;
    initialize_pressure(&pressure);

    // Set exact pressure at t=0
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                pressure.p[idx] = p_exact_value(x, y, z, 0.0);
            }
        }
    }

    // Initialize the 3 velocity fields
    VelocityField Eta, Zeta, U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

    // Set exact initial velocity at t=0
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                
                U.v_x[idx] = u_exact_component(x, y, z, 0.0, 0);
                U.v_y[idx] = u_exact_component(x, y, z, 0.0, 1);
                U.v_z[idx] = u_exact_component(x, y, z, 0.0, 2);
                
                Eta.v_x[idx] = U.v_x[idx];
                Eta.v_y[idx] = U.v_y[idx];
                Eta.v_z[idx] = U.v_z[idx];
                
                Zeta.v_x[idx] = U.v_x[idx];
                Zeta.v_y[idx] = U.v_y[idx];
                Zeta.v_z[idx] = U.v_z[idx];
            }
        }
    }

    // Set K (permeability)
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE);
    for (size_t i = 0; i < GRID_ELEMENTS; i++) {
        K[i] = 1.0;
    }

    // Compute Beta and Gamma
    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE);
    for (size_t idx = 0; idx < GRID_ELEMENTS; idx++) {
        Beta[idx] = 1.0 + (DT * NU) / (2.0 * K[idx]);
        Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
    }

    // Initialize G field
    GField g_field;
    initialize_g_field(&g_field);

    // Boundary conditions
    DTYPE *u_BC_current = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));
    DTYPE *u_BC_second = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));
    DTYPE *u_BC_third = (DTYPE *)calloc(GRID_ELEMENTS, sizeof(DTYPE));

    // Allocate arrays to record solution at each timestep
    VelocityField *U_record = (VelocityField *)malloc((STEPS) * sizeof(VelocityField));
    Pressure *P_record = (Pressure *)malloc((STEPS) * sizeof(Pressure));
    
    for (int t = 0; t < STEPS; t++) {
        initialize_velocity_field(&U_record[t]);
        initialize_pressure(&P_record[t]);
    }

    // Run solver with manufactured forcing - record all timesteps
    int write_frequency = 1;
    bool full_output = true;
    solve(g_field, manufactured_forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, 
          u_BC_current, u_BC_second, u_BC_third, write_frequency, full_output, &U_record, &P_record);

    // Verify solution at final time using recorded arrays
    double t_final = (STEPS - 1) * DT;
    
    // Error metrics
    double max_error_abs = 0.0;      // L∞ norm (maximum absolute error)
    double sum_error_sq = 0.0;       // For L2 norm
    double sum_exact_sq = 0.0;       // For relative L2 norm
    int n_interior_points = 0;
    
    std::cout << "\n[DEBUG] Computing errors at t_final = " << t_final << std::endl;
    std::cout << "[DEBUG] Using recorded timestep index: " << (STEPS - 1) << std::endl;
    
    for (int k = 1; k < DEPTH-1; k++) {
        for (int j = 1; j < HEIGHT-1; j++) {
            for (int i = 1; i < WIDTH-1; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                
                // Exact solution at final time
                double ux_exact = u_exact_component(x, y, z, t_final, 0);
                double uy_exact = u_exact_component(x, y, z, t_final, 1);
                double uz_exact = u_exact_component(x, y, z, t_final, 2);
                
                // Numerical solution from recorded array (last timestep)
                double ux_num = U_record[STEPS - 1].v_x[idx];
                double uy_num = U_record[STEPS - 1].v_y[idx];
                double uz_num = U_record[STEPS - 1].v_z[idx];
                
                // Absolute errors per component
                double error_x = fabs(ux_num - ux_exact);
                double error_y = fabs(uy_num - uy_exact);
                double error_z = fabs(uz_num - uz_exact);
                
                // L∞ norm: maximum component-wise error
                double point_max_error = fmax(error_x, fmax(error_y, error_z));
                max_error_abs = fmax(max_error_abs, point_max_error);
                
                // Accumulate for L2 norm
                sum_error_sq += error_x * error_x + error_y * error_y + error_z * error_z;
                sum_exact_sq += ux_exact * ux_exact + uy_exact * uy_exact + uz_exact * uz_exact;
                
                n_interior_points++;
            }
        }
    }
    
    // Compute L2 norm: sqrt(sum(error²) / N)
    double error_L2 = sqrt(sum_error_sq / n_interior_points);
    
    // Compute magnitude of exact solution: sqrt(sum(exact²) / N)
    double exact_L2 = sqrt(sum_exact_sq / n_interior_points);
    
    // Relative L2 error: ||error||_L2 / ||exact||_L2
    double error_L2_relative = error_L2 / exact_L2;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Manufactured Solution Error Analysis  " << std::endl;
    std::cout << "  (From Memory - U_record/P_record)     " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Time:                    t = " << std::fixed << std::setprecision(6) << t_final << std::endl;
    std::cout << "Interior points:         N = " << n_interior_points << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "L∞ norm (max abs error): " << std::scientific << std::setprecision(6) << max_error_abs << std::endl;
    std::cout << "L2 norm (RMS error):     " << std::scientific << std::setprecision(6) << error_L2 << std::endl;
    std::cout << "L2 exact (RMS solution): " << std::scientific << std::setprecision(6) << exact_L2 << std::endl;
    std::cout << "Relative L2 error:       " << std::scientific << std::setprecision(6) 
              << error_L2_relative << " (" << std::fixed << std::setprecision(2) 
              << error_L2_relative * 100.0 << "%)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Expect convergence - use both L∞ and relative L2 error
    EXPECT_LT(error_L2_relative, 0.1) 
        << "Relative L2 error should be < 10%";
    EXPECT_LT(max_error_abs, 1.0) 
        << "Maximum absolute error (L∞) should be reasonable";

    // Cleanup
    for (int t = 0; t < STEPS; t++) {
        free_velocity_field(&U_record[t]);
        free_pressure(&P_record[t]);
    }
    free(U_record);
    free(P_record);
    
    free(K);
    free(Beta);
    free(Gamma);
    free(u_BC_current);
    free(u_BC_second);
    free(u_BC_third);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);
}





