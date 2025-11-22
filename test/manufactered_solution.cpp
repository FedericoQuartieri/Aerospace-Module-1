#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <string>

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
    // Last timestep is always STEPS since we write every step
    int last_step = STEPS;
    
    std::stringstream filename;
    filename << "output/velocity_field_" << last_step << ".vtk";
    
    std::ifstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename.str() << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header lines until we find POINT_DATA
    while (std::getline(file, line)) {
        if (line.find("POINT_DATA") != std::string::npos) {
            break;
        }
    }
    
    // Read velocity components
    std::getline(file, line); // VECTORS Velocity double
    for (size_t idx = 0; idx < GRID_SIZE; idx++) {
        file >> U_numerical.v_x[idx] >> U_numerical.v_y[idx] >> U_numerical.v_z[idx];
    }
    
    // Skip to pressure data
    std::getline(file, line); // empty line
    std::getline(file, line); // SCALARS Pressure double
    std::getline(file, line); // LOOKUP_TABLE default
    
    for (size_t idx = 0; idx < GRID_SIZE; idx++) {
        file >> P_numerical.p[idx];
    }
    
    file.close();
    return true;
}


TEST(ManufacturedSolution, VelocitySystemConvergence)
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
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE * sizeof(DTYPE));
    for (size_t i = 0; i < GRID_SIZE; i++) {
        K[i] = 1.0; // Uniform permeability
    }

    // Compute Beta and Gamma
    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE * sizeof(DTYPE));
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE * sizeof(DTYPE));
    for (size_t idx = 0; idx < GRID_SIZE; idx++) {
        Beta[idx] = 1.0 + (DT * NU) / (2.0 * K[idx]);
        Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
    }

    // Initialize G field
    GField g_field;
    initialize_g_field(&g_field);

    // Boundary conditions (can be zero or exact solution at boundaries)
    DTYPE *u_BC_current = (DTYPE *)calloc(GRID_SIZE, sizeof(DTYPE));
    DTYPE *u_BC_second = (DTYPE *)calloc(GRID_SIZE, sizeof(DTYPE));
    DTYPE *u_BC_third = (DTYPE *)calloc(GRID_SIZE, sizeof(DTYPE));

    // Run solver with manufactured forcing - write every timestep for accuracy
    int write_frequency = 1;
    solve(g_field, manufactured_forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, 
          u_BC_current, u_BC_second, u_BC_third, write_frequency);

    // Read numerical solution from last VTK file
    VelocityField U_numerical;
    Pressure P_numerical;
    initialize_velocity_field(&U_numerical);
    initialize_pressure(&P_numerical);
    
    ASSERT_TRUE(read_last_vtk_file(U_numerical, P_numerical)) << "Failed to read VTK output file";

    // Verify solution at final time
    double t_final = STEPS * DT;
    double max_error = 0.0;
    
    for (int k = 1; k < DEPTH-1; k++) {
        for (int j = 1; j < HEIGHT-1; j++) {
            for (int i = 1; i < WIDTH-1; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                double x = i * DX;
                double y = j * DY;
                double z = k * DZ;
                
                double ux_exact = u_exact_component(x, y, z, t_final, 0);
                double uy_exact = u_exact_component(x, y, z, t_final, 1);
                double uz_exact = u_exact_component(x, y, z, t_final, 2);
                
                double error_x = fabs(U_numerical.v_x[idx] - ux_exact);
                double error_y = fabs(U_numerical.v_y[idx] - uy_exact);
                double error_z = fabs(U_numerical.v_z[idx] - uz_exact);
                
                max_error = fmax(max_error, fmax(error_x, fmax(error_y, error_z)));
            }
        }
    }

    printf("Maximum error at t_final = %f: %e\n", t_final, max_error);
    
    // Expect convergence (adjust tolerance based on grid resolution and time step)
    EXPECT_LT(max_error, 0.1); // Adjust tolerance as needed

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

