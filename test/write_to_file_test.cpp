#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include "utils.hpp"


extern "C"
{
#include "g_field.h"
#include "velocity_field.h"
#include "pressure.h"
#include "solve.h"
#include "forcing_parser.h"
#include <math.h>
}



TEST(ManufacturedSolution, FileMemoryConsistency)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "  File vs Memory Consistency Test      " << std::endl;
    std::cout << "========================================\n" << std::endl;

    function forcing;
    double x, y, z, t;
    double fx, fy, fz;

    forcing = parse_forcing_function("../forcing.txt");
    if (!forcing) {
        /* Error already printed to stderr */
        throw std::runtime_error("Failed to parse forcing function");
    }



    // Initialize pressure
    Pressure pressure;
    initialize_pressure(&pressure);
    rand_fill_pressure(&pressure); // Random initial pressure

    // Initialize the 3 velocity fields
    VelocityField Eta, Zeta, U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

    rand_fill_velocity_field(&U); // Random initial velocity
    rand_fill_velocity_field(&Eta); // Random initial velocity
    rand_fill_velocity_field(&Zeta); // Random initial velocity
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

    // Run solver - ENABLE BOTH file writing AND memory recording
    int write_frequency = 1;
    bool full_output = true;
    solve(g_field, forcing, pressure, K, Eta, Zeta, U, Beta, Gamma, 
          u_BC_current, u_BC_second, u_BC_third, write_frequency, full_output, &U_record, &P_record);

    std::cout << "[INFO] Solver completed. Now comparing file vs memory data...\n" << std::endl;

    // Read numerical solution from last VTK file
    VelocityField U_from_file;
    Pressure P_from_file;
    initialize_velocity_field(&U_from_file);
    initialize_pressure(&P_from_file);
    
    ASSERT_TRUE(read_last_vtk_file(U_from_file, P_from_file)) 
        << "Failed to read VTK output file";

    // Compare file data vs memory data
    double max_diff_vx = 0.0;
    double max_diff_vy = 0.0;
    double max_diff_vz = 0.0;
    double max_diff_p = 0.0;
    
    double sum_diff_sq_vx = 0.0;
    double sum_diff_sq_vy = 0.0;
    double sum_diff_sq_vz = 0.0;
    double sum_diff_sq_p = 0.0;
    
    int total_points = WIDTH * HEIGHT * DEPTH;
    
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                
                // Velocity differences
                double diff_vx = fabs(U_from_file.v_x[idx] - U_record[STEPS - 1].v_x[idx]);
                double diff_vy = fabs(U_from_file.v_y[idx] - U_record[STEPS - 1].v_y[idx]);
                double diff_vz = fabs(U_from_file.v_z[idx] - U_record[STEPS - 1].v_z[idx]);
                
                // Pressure difference
                double diff_p = fabs(P_from_file.p[idx] - P_record[STEPS - 1].p[idx]);
                
                // Update max differences
                max_diff_vx = fmax(max_diff_vx, diff_vx);
                max_diff_vy = fmax(max_diff_vy, diff_vy);
                max_diff_vz = fmax(max_diff_vz, diff_vz);
                max_diff_p = fmax(max_diff_p, diff_p);
                
                // Accumulate for RMS
                sum_diff_sq_vx += diff_vx * diff_vx;
                sum_diff_sq_vy += diff_vy * diff_vy;
                sum_diff_sq_vz += diff_vz * diff_vz;
                sum_diff_sq_p += diff_p * diff_p;
            }
        }
    }
    
    // Compute RMS differences
    double rms_diff_vx = sqrt(sum_diff_sq_vx / total_points);
    double rms_diff_vy = sqrt(sum_diff_sq_vy / total_points);
    double rms_diff_vz = sqrt(sum_diff_sq_vz / total_points);
    double rms_diff_p = sqrt(sum_diff_sq_p / total_points);
    
    // Print comparison results
    std::cout << "========================================" << std::endl;
    std::cout << "  File vs Memory Consistency Results   " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total grid points: " << total_points << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "MAX DIFFERENCES (L∞ norm):" << std::endl;
    std::cout << "  Velocity_x: " << std::scientific << std::setprecision(6) << max_diff_vx << std::endl;
    std::cout << "  Velocity_y: " << std::scientific << std::setprecision(6) << max_diff_vy << std::endl;
    std::cout << "  Velocity_z: " << std::scientific << std::setprecision(6) << max_diff_vz << std::endl;
    std::cout << "  Pressure:   " << std::scientific << std::setprecision(6) << max_diff_p << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "RMS DIFFERENCES (L2 norm):" << std::endl;
    std::cout << "  Velocity_x: " << std::scientific << std::setprecision(6) << rms_diff_vx << std::endl;
    std::cout << "  Velocity_y: " << std::scientific << std::setprecision(6) << rms_diff_vy << std::endl;
    std::cout << "  Velocity_z: " << std::scientific << std::setprecision(6) << rms_diff_vz << std::endl;
    std::cout << "  Pressure:   " << std::scientific << std::setprecision(6) << rms_diff_p << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Consistency checks - data should be IDENTICAL (within machine precision)
    const double tolerance = 1e-14;  // Machine precision tolerance
    
    EXPECT_LT(max_diff_vx, tolerance) 
        << "File and memory data for Velocity_x should be identical";
    EXPECT_LT(max_diff_vy, tolerance) 
        << "File and memory data for Velocity_y should be identical";
    EXPECT_LT(max_diff_vz, tolerance) 
        << "File and memory data for Velocity_z should be identical";
    EXPECT_LT(max_diff_p, tolerance) 
        << "File and memory data for Pressure should be identical";
    
    EXPECT_LT(rms_diff_vx, tolerance) 
        << "RMS difference for Velocity_x should be negligible";
    EXPECT_LT(rms_diff_vy, tolerance) 
        << "RMS difference for Velocity_y should be negligible";
    EXPECT_LT(rms_diff_vz, tolerance) 
        << "RMS difference for Velocity_z should be negligible";
    EXPECT_LT(rms_diff_p, tolerance) 
        << "RMS difference for Pressure should be negligible";

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
    free_pressure(&P_from_file);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_velocity_field(&U_from_file);
    free_g_field(&g_field);
}








