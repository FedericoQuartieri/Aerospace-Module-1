#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"

// Include C headers in extern "C" block for C++ compatibility
extern "C" {
#include "g_field.h"
#include "velocity_field.h"
#include "pressure.h"
#include "forcing_parser.h"
}

/**
 * Test Suite: GFieldComputationTest
 * 
 * This suite verifies the compute_g function which implements the 
 * Navier-Stokes-Brinkman equation for computing the G field:
 * 
 * g^(n+1/2) = f^(n+1/2) - ∇p*^(n+1/2) - (ν/2k)u^n + (ν/2)(∂_xx η^n + ∂_yy ζ^n + ∂_zz u^n)
 */

// Global force values for testing (will be set by each test)
static double test_force_x = 0.0;
static double test_force_y = 0.0;
static double test_force_z = 0.0;

// Simple forcing function for tests
static double test_forcing(double x, double y, double z, double t, int component) {
    (void)x; (void)y; (void)z; (void)t; // Unused parameters
    switch(component) {
        case 0: return test_force_x;
        case 1: return test_force_y;
        case 2: return test_force_z;
        default: return 0.0;
    }
}

// Helper function to allocate and initialize all fields with known values
void setup_test_fields(GField* g_field, Pressure* pressure,
                       DTYPE** K, VelocityField* Eta, VelocityField* Zeta, 
                       VelocityField* U, bool initialize_to_zero = true) {
    // Allocate G field
    initialize_g_field(g_field);
    
    // Allocate pressure
    initialize_pressure(pressure);
    
    // Allocate permeability K
    *K = (DTYPE*) malloc(TOTAL_GRID_POINTS * sizeof(DTYPE));
    
    // Allocate velocity fields
    initialize_velocity_field(Eta);
    initialize_velocity_field(Zeta);
    initialize_velocity_field(U);
    
    if (initialize_to_zero) {
        // Initialize all fields to zero
        memset(g_field->g_x, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(g_field->g_y, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(g_field->g_z, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(pressure->p, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(*K, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Eta->v_x, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Eta->v_y, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Eta->v_z, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Zeta->v_x, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Zeta->v_y, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(Zeta->v_z, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(U->v_x, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(U->v_y, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
        memset(U->v_z, 0, TOTAL_GRID_POINTS * sizeof(DTYPE));
    }
}

// Helper to cleanup all fields
void cleanup_test_fields(GField* g_field, Pressure* pressure,
                        DTYPE* K, VelocityField* Eta, VelocityField* Zeta, 
                        VelocityField* U) {
    free_g_field(g_field);
    free_pressure(pressure);
    free(K);
    free_velocity_field(Eta);
    free_velocity_field(Zeta);
    free_velocity_field(U);
}

TEST(GFieldComputationTest, SinglePointXComponent) {
    // Test G_x computation at a single interior point
    GField g_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Set test forcing values
    test_force_x = 10.0;
    test_force_y = 0.0;
    test_force_z = 0.0;
    
    // Choose an interior test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    
    K[idx] = 1.0;
    U.v_x[idx] = 0.0;
    
    // Run compute_g with time_step = 0
    compute_g(&g_field, test_forcing, &pressure, K, &Eta, &Zeta, &U, 0);
    
    // Verify G_x at the test point
    EXPECT_NEAR(g_field.g_x[idx], 10.0, 1e-12)
        << "G_x should equal f_x when all other terms are zero";
    
    cleanup_test_fields(&g_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, SinglePointYComponent) {
    // Test G_y computation at a single interior point with pressure gradient
    GField g_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Set test forcing values
    test_force_x = 0.0;
    test_force_y = 10000.0;
    test_force_z = 0.0;
    
    // Test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    size_t idx_y_plus = rowmaj_idx(i, j+1, k);
    
    K[idx] = 1.0;
    U.v_y[idx] = 0.0;
    
    // Set pressure gradient in y
    pressure.p[idx] = 5.0;
    pressure.p[idx_y_plus] = 10.0;

    compute_g(&g_field, test_forcing, &pressure, K, &Eta, &Zeta, &U, 0);

    EXPECT_NEAR(g_field.g_y[idx], 5000.0, 1e-12)
        << "G_y should be f_y minus pressure gradient";
    
    cleanup_test_fields(&g_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, SinglePointZComponent) {
    // Test G_z computation with Brinkman damping term
    GField g_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Set test forcing values
    test_force_x = 0.0;
    test_force_y = 0.0;
    test_force_z = 50.0;
    
    // Test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    
    K[idx] = 2.0;
    U.v_z[idx] = 10.0;
    U.v_z[rowmaj_idx(i, j, k-1)] = 10.0;
    U.v_z[rowmaj_idx(i, j, k+1)] = 10.0;
    
    DTYPE expected_brinkman = -(NU / (2.0 * K[idx])) * U.v_z[idx];
    DTYPE expected_g_z = 50.0 + expected_brinkman;
    
    compute_g(&g_field, test_forcing, &pressure, K, &Eta, &Zeta, &U, 0);
    
    EXPECT_NEAR(g_field.g_z[idx], expected_g_z, 1e-12)
        << "G_z should include Brinkman damping term";
    
    cleanup_test_fields(&g_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, FullField10x10x10AllComponents) {
    GField g_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &pressure, &K, &Eta, &Zeta, &U, false);
    
    // Set constant forcing
    test_force_x = 1.0;
    test_force_y = 2.0;
    test_force_z = 3.0;
    
    // Initialize fields with known linear distributions
    for(int kk = 0; kk < DEPTH; kk++) {
        for(int jj = 0; jj < HEIGHT; jj++) {
            for(int ii = 0; ii < WIDTH; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);

                // Pressure: linear p = 10 + 2*i + 3*j + 4*k
                pressure.p[idx] = 10.0 + 2.0*ii + 3.0*jj + 4.0*kk;

                // Permeability: constant
                K[idx] = 1.0;

                // Velocity fields: constant
                U.v_x[idx] = 5.0;
                U.v_y[idx] = 6.0;
                U.v_z[idx] = 7.0;

                Eta.v_x[idx] = 1.0;
                Eta.v_y[idx] = 1.0;
                Eta.v_z[idx] = 1.0;

                Zeta.v_x[idx] = 1.0;
                Zeta.v_y[idx] = 1.0;
                Zeta.v_z[idx] = 1.0;
            }
        }
    }
    
    // Run compute_g
    compute_g(&g_field, test_forcing, &pressure, K, &Eta, &Zeta, &U, 0);
    
    // Verify interior points
    int points_checked = 0;
    
    for(int kk = 1; kk < DEPTH-1; kk++) {
        for(int jj = 1; jj < HEIGHT-1; jj++) {
            for(int ii = 1; ii < WIDTH-1; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                DTYPE grad_p_x = 2.0 * DX_INVERSE;
                DTYPE grad_p_y = 3.0 * DY_INVERSE;
                DTYPE grad_p_z = 4.0 * DZ_INVERSE;

                DTYPE brinkman_x = -(NU / (2.0 * K[idx])) * U.v_x[idx];
                DTYPE brinkman_y = -(NU / (2.0 * K[idx])) * U.v_y[idx];
                DTYPE brinkman_z = -(NU / (2.0 * K[idx])) * U.v_z[idx];

                DTYPE viscous_x = 0.0;
                DTYPE viscous_y = 0.0;
                DTYPE viscous_z = 0.0;

                DTYPE expected_g_x = test_force_x - grad_p_x + brinkman_x + viscous_x;
                DTYPE expected_g_y = test_force_y - grad_p_y + brinkman_y + viscous_y;
                DTYPE expected_g_z = test_force_z - grad_p_z + brinkman_z + viscous_z;
                
                EXPECT_NEAR(g_field.g_x[idx], expected_g_x, 1e-10)
                    << "G_x mismatch at (" << ii << "," << jj << "," << kk << ")";
                    
                EXPECT_NEAR(g_field.g_y[idx], expected_g_y, 1e-10)
                    << "G_y mismatch at (" << ii << "," << jj << "," << kk << ")";
                    
                EXPECT_NEAR(g_field.g_z[idx], expected_g_z, 1e-10)
                    << "G_z mismatch at (" << ii << "," << jj << "," << kk << ")";
                
                points_checked++;
            }
        }
    }
    
    EXPECT_EQ(points_checked, (WIDTH-2) * (HEIGHT-2) * (DEPTH-2))
        << "Should check all interior points";
    
    cleanup_test_fields(&g_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, ComplexFullFieldWithViscousTerms) {
    GField g_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &pressure, &K, &Eta, &Zeta, &U, false);
    
    // Set constant forcing
    test_force_x = 10.0;
    test_force_y = 20.0;
    test_force_z = 30.0;
    
    const DTYPE coeff_u_x = 0.1;
    const DTYPE coeff_u_y = 0.2;
    const DTYPE coeff_u_z = 0.3;
    const DTYPE coeff_eta_x = 0.05;
    const DTYPE coeff_zeta_y = 0.05;
    
    for(int kk = 0; kk < DEPTH; kk++) {
        for(int jj = 0; jj < HEIGHT; jj++) {
            for(int ii = 0; ii < WIDTH; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                pressure.p[idx] = 100.0 + 5.0*ii + 6.0*jj + 7.0*kk;
                K[idx] = 2.0;
                
                U.v_x[idx] = coeff_u_x * ii * ii;
                U.v_y[idx] = coeff_u_y * jj * jj;
                U.v_z[idx] = coeff_u_z * kk * kk;
                
                Eta.v_x[idx] = coeff_eta_x * ii * ii;
                Eta.v_y[idx] = 0.0;
                Eta.v_z[idx] = 0.0;
                
                Zeta.v_x[idx] = 0.0;
                Zeta.v_y[idx] = coeff_zeta_y * jj * jj;
                Zeta.v_z[idx] = 0.0;
            }
        }
    }
    
    compute_g(&g_field, test_forcing, &pressure, K, &Eta, &Zeta, &U, 0);
    
    int points_checked = 0;
    
    for(int kk = 1; kk < DEPTH-1; kk++) {
        for(int jj = 1; jj < HEIGHT-1; jj++) {
            for(int ii = 1; ii < WIDTH-1; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                DTYPE grad_p_x = 5.0 * DX_INVERSE;
                DTYPE grad_p_y = 6.0 * DY_INVERSE;
                DTYPE grad_p_z = 7.0 * DZ_INVERSE;
                
                DTYPE brinkman_x = -(NU / (2.0 * K[idx])) * U.v_x[idx];
                DTYPE brinkman_y = -(NU / (2.0 * K[idx])) * U.v_y[idx];
                DTYPE brinkman_z = -(NU / (2.0 * K[idx])) * U.v_z[idx];
                
                DTYPE d2_eta_x_dx2 = 2.0 * coeff_eta_x * DX_INVERSE_SQUARE;
                DTYPE d2_zeta_y_dy2 = 2.0 * coeff_zeta_y * DY_INVERSE_SQUARE;
                DTYPE d2_u_z_dz2 = 2.0 * coeff_u_z * DZ_INVERSE_SQUARE;

                DTYPE viscous_x = (NU / 2.0) * d2_eta_x_dx2;
                DTYPE viscous_y = (NU / 2.0) * d2_zeta_y_dy2;
                DTYPE viscous_z = (NU / 2.0) * d2_u_z_dz2;
                
                DTYPE expected_g_x = test_force_x - grad_p_x + brinkman_x + viscous_x;
                DTYPE expected_g_y = test_force_y - grad_p_y + brinkman_y + viscous_y;
                DTYPE expected_g_z = test_force_z - grad_p_z + brinkman_z + viscous_z;
                
                EXPECT_NEAR(g_field.g_x[idx], expected_g_x, 1e-6)
                    << "G_x mismatch at (" << ii << "," << jj << "," << kk << ")";
                
                EXPECT_NEAR(g_field.g_y[idx], expected_g_y, 1e-6)
                    << "G_y mismatch at (" << ii << "," << jj << "," << kk << ")";
                
                EXPECT_NEAR(g_field.g_z[idx], expected_g_z, 1e-6)
                    << "G_z mismatch at (" << ii << "," << jj << "," << kk << ")";
                
                points_checked++;
            }
        }
    }
    
    EXPECT_EQ(points_checked, (WIDTH-2) * (HEIGHT-2) * (DEPTH-2))
        << "Should check all interior points";
    
    cleanup_test_fields(&g_field, &pressure, K, &Eta, &Zeta, &U);
}
