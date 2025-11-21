#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"

// Include C headers in extern "C" block for C++ compatibility
extern "C" {
#include "g_field.h"
#include "force_field.h"
#include "velocity_field.h"
#include "pressure.h"
}

/**
 * Test Suite: GFieldComputationTest
 * 
 * This suite verifies the compute_g function which implements the 
 * Navier-Stokes-Brinkman equation for computing the G field:
 * 
 * g^(n+1/2) = f^(n+1/2) - ∇p*^(n+1/2) - (ν/2k)u^n + (ν/2)(∂_xx η^n + ∂_yy ζ^n + ∂_zz u^n)
 * 
 * where:
 *   - f is the external force field
 *   - p* is the pressure field
 *   - k is the permeability field
 *   - u, η (Eta), ζ (Zeta) are velocity fields
 *   - ν is the kinematic viscosity (NU)
 */

// Helper function to allocate and initialize all fields with known values
void setup_test_fields(GField* g_field, ForceField* f_field, Pressure* pressure,
                       DTYPE** K, VelocityField* Eta, VelocityField* Zeta, 
                       VelocityField* U, bool initialize_to_zero = true) {
    // Allocate G field
    g_field->g_x = (DTYPE*) malloc(GRID_SIZE);
    g_field->g_y = (DTYPE*) malloc(GRID_SIZE);
    g_field->g_z = (DTYPE*) malloc(GRID_SIZE);
    
    // Allocate force field
    f_field->f_x = (DTYPE*) malloc(GRID_SIZE);
    f_field->f_y = (DTYPE*) malloc(GRID_SIZE);
    f_field->f_z = (DTYPE*) malloc(GRID_SIZE);
    
    // Allocate pressure
    pressure->p = (DTYPE*) malloc(GRID_SIZE);
    
    // Allocate permeability K
    *K = (DTYPE*) malloc(GRID_SIZE);
    
    // Allocate velocity fields
    Eta->v_x = (DTYPE*) malloc(GRID_SIZE);
    Eta->v_y = (DTYPE*) malloc(GRID_SIZE);
    Eta->v_z = (DTYPE*) malloc(GRID_SIZE);
    
    Zeta->v_x = (DTYPE*) malloc(GRID_SIZE);
    Zeta->v_y = (DTYPE*) malloc(GRID_SIZE);
    Zeta->v_z = (DTYPE*) malloc(GRID_SIZE);
    
    U->v_x = (DTYPE*) malloc(GRID_SIZE);
    U->v_y = (DTYPE*) malloc(GRID_SIZE);
    U->v_z = (DTYPE*) malloc(GRID_SIZE);
    
    if (initialize_to_zero) {
        // Initialize all fields to zero
        memset(g_field->g_x, 0, GRID_SIZE);
        memset(g_field->g_y, 0, GRID_SIZE);
        memset(g_field->g_z, 0, GRID_SIZE);
        memset(f_field->f_x, 0, GRID_SIZE);
        memset(f_field->f_y, 0, GRID_SIZE);
        memset(f_field->f_z, 0, GRID_SIZE);
        memset(pressure->p, 0, GRID_SIZE);
        memset(*K, 0, GRID_SIZE);
        memset(Eta->v_x, 0, GRID_SIZE);
        memset(Eta->v_y, 0, GRID_SIZE);
        memset(Eta->v_z, 0, GRID_SIZE);
        memset(Zeta->v_x, 0, GRID_SIZE);
        memset(Zeta->v_y, 0, GRID_SIZE);
        memset(Zeta->v_z, 0, GRID_SIZE);
        memset(U->v_x, 0, GRID_SIZE);
        memset(U->v_y, 0, GRID_SIZE);
        memset(U->v_z, 0, GRID_SIZE);
    }
}

// Helper to cleanup all fields
void cleanup_test_fields(GField* g_field, ForceField* f_field, Pressure* pressure,
                        DTYPE* K, VelocityField* Eta, VelocityField* Zeta, 
                        VelocityField* U) {
    free(g_field->g_x);
    free(g_field->g_y);
    free(g_field->g_z);
    free(f_field->f_x);
    free(f_field->f_y);
    free(f_field->f_z);
    free(pressure->p);
    free(K);
    free(Eta->v_x);
    free(Eta->v_y);
    free(Eta->v_z);
    free(Zeta->v_x);
    free(Zeta->v_y);
    free(Zeta->v_z);
    free(U->v_x);
    free(U->v_y);
    free(U->v_z);
}

TEST(GFieldComputationTest, SinglePointXComponent) {
    // Test G_x computation at a single interior point
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Choose an interior test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    
    // Set up simple test case: only force field has non-zero value
    // G_x should equal f_x when all other terms are zero
    f_field.f_x[idx] = 10.0;
    // Ensure permeability K is non-zero to avoid division-by-zero in compute_g
    // (compute_g uses NU / (2.0 * K[idx]) ). Set K to 1.0 at this index.
    K[idx] = 1.0;
    // Ensure velocity U is zero at this index so the Brinkman term contributes 0.0
    U.v_x[idx] = 0.0;
    
    // All pressure gradients, velocity terms, and K are zero
    // Expected: g_x = f_x - 0 - 0 + 0 = 10.0
    
    // Run compute_g
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
    
    // Verify G_x at the test point
    EXPECT_NEAR(g_field.g_x[idx], 10.0, 1e-12)
        << "G_x should equal f_x when all other terms are zero";
    
    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, SinglePointYComponent) {
    // Test G_y computation at a single interior point with pressure gradient
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    size_t idx_y_plus = rowmaj_idx(i, j+1, k);
    
    // Set force field
    f_field.f_y[idx] = 10000.0;

    K[idx] = 1.0; // Set K to 1.0 to avoid division by zero
    
    U.v_y[idx] = 0.0; // Set U to 0.
    
    // Set pressure gradient in y: p[j] = 5.0, p[j+1] = 10.0
    // grad_y = (10.0 - 5.0) / 0.1 = 50.0
    pressure.p[idx] = 5.0;
    pressure.p[idx_y_plus] = 10.0;

    // Expected: g_y = f_y - grad_y(P) - 0 + 0 = 10000.0 - 5000.0 = 5000.0

    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);

    EXPECT_NEAR(g_field.g_y[idx], 5000.0, 1e-12)
        << "G_y should be f_y minus pressure gradient";
    
    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, SinglePointZComponent) {
    // Test G_z computation with Brinkman damping term
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U);
    
    // Test point (i=2, j=2, k=2)
    size_t i = 2, j = 2, k = 2;
    size_t idx = rowmaj_idx(i, j, k);
    
    // Set force field
    f_field.f_z[idx] = 50.0;
    
    // Set permeability K and velocity U_z to test Brinkman term
    // Brinkman term: -(NU / (2 * K)) * U_z = -(0.7 / (2 * 2.0)) * 10.0 = -(0.7 / 4.0) * 10.0 = -0.175 * 10.0 = -1.75
    K[idx] = 2.0;
    // IMPORTANT: compute_g adds a viscous term that contains second derivatives
    // of the velocity fields (compute_velocity_zz_grad). If we only set a
    // single point in U.v_z to a non-zero value, the second derivative will
    // be non-zero because neighbors are zero. To isolate the Brinkman term
    // we set the value on the point and its immediate neighbors in z so
    // the second derivative becomes zero (constant in the local stencil).
    U.v_z[idx] = 10.0;
    U.v_z[rowmaj_idx(i, j, k-1)] = 10.0;
    U.v_z[rowmaj_idx(i, j, k+1)] = 10.0;
    
    // Expected: g_z = f_z - 0 - (NU / (2*K)) * U_z + 0
    //                = 50.0 - 0 - 1.75 + 0 = 48.25
    DTYPE expected_brinkman = -(NU / (2.0 * K[idx])) * U.v_z[idx];
    DTYPE expected_g_z = f_field.f_z[idx] + expected_brinkman;
    
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
    
    EXPECT_NEAR(g_field.g_z[idx], expected_g_z, 1e-12)
        << "G_z should include Brinkman damping term";
    
    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, FullField10x10x10AllComponents) {
    // Test all three components of G over a complete 10x10x10 field
    // with all terms (force, pressure gradient, Brinkman, viscous diffusion)
    
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, false);
    
    // Initialize fields with known linear distributions on the project grid
    // (use WIDTH/HEIGHT/DEPTH from constants.h). This lets us compute expected
    // gradients from the project's DX/DY/DZ values.
    for(int kk = 0; kk < DEPTH; kk++) {
        for(int jj = 0; jj < HEIGHT; jj++) {
            for(int ii = 0; ii < WIDTH; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);

                // Force field: constant values
                f_field.f_x[idx] = 1.0;
                f_field.f_y[idx] = 2.0;
                f_field.f_z[idx] = 3.0;

                // Pressure: linear in all directions p = 10 + 2*i + 3*j + 4*k
                // Gradients will be computed as 2/DX, 3/DY, 4/DZ
                pressure.p[idx] = 10.0 + 2.0*ii + 3.0*jj + 4.0*kk;

                // Permeability: constant (avoid zero)
                K[idx] = 1.0;

                // Velocity fields: constant (so second derivatives are zero)
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
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
    
    // Verify all interior points (excluding boundaries where forward differences would fail)
    // Loop from i=1 to WIDTH-2, j=1 to HEIGHT-2, k=1 to DEPTH-2
    int points_checked = 0;
    
    for(int kk = 1; kk < DEPTH-1; kk++) {
        for(int jj = 1; jj < HEIGHT-1; jj++) {
            for(int ii = 1; ii < WIDTH-1; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                // Expected values computed from project constants
                // Pressure gradients (constant): grad_x = 2.0/DX, grad_y = 3.0/DY, grad_z = 4.0/DZ
                DTYPE grad_p_x = 2.0 * DX_INVERSE;
                DTYPE grad_p_y = 3.0 * DY_INVERSE;
                DTYPE grad_p_z = 4.0 * DZ_INVERSE;

                // Brinkman term: -(NU / (2*K)) * U (K[idx] set to 1.0 above)
                DTYPE brinkman_x = -(NU / (2.0 * K[idx])) * U.v_x[idx];
                DTYPE brinkman_y = -(NU / (2.0 * K[idx])) * U.v_y[idx];
                DTYPE brinkman_z = -(NU / (2.0 * K[idx])) * U.v_z[idx];

                // Viscous diffusion: velocities are constant, so all second derivatives = 0
                DTYPE viscous_x = 0.0;
                DTYPE viscous_y = 0.0;
                DTYPE viscous_z = 0.0;

                // Expected G values (use the same algebra as compute_g)
                DTYPE expected_g_x = f_field.f_x[idx] - grad_p_x + brinkman_x + viscous_x;
                DTYPE expected_g_y = f_field.f_y[idx] - grad_p_y + brinkman_y + viscous_y;
                DTYPE expected_g_z = f_field.f_z[idx] - grad_p_z + brinkman_z + viscous_z;
                
                // Verify each component
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
    
    // Verify we checked a reasonable number of interior points
    // For 10x10x10 grid, interior is 8x8x8 = 512 points (excluding boundaries)
    EXPECT_EQ(points_checked, (WIDTH-2) * (HEIGHT-2) * (DEPTH-2))
        << "Should check all interior points";
    
    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
}

TEST(GFieldComputationTest, ComplexFullFieldWithViscousTerms) {
    /**
     * More complex test with NON-ZERO viscous diffusion terms.
     * 
     * Strategy:
     * - Use quadratic velocity fields so second derivatives are non-zero and constant
     * - Example: U.v_x = a*i^2 => d²U_x/di² = 2a (constant second derivative)
     * - Verify all terms: force, pressure gradient, Brinkman, AND viscous diffusion
     * 
     * Velocity field setup:
     *   U.v_x(i,j,k) = 0.1*i^2  =>  ∂²U_x/∂x² = 0.2 * DX_INVERSE^2
     *   U.v_y(i,j,k) = 0.2*j^2  =>  ∂²U_y/∂y² = 0.4 * DY_INVERSE^2
     *   U.v_z(i,j,k) = 0.3*k^2  =>  ∂²U_z/∂z² = 0.6 * DZ_INVERSE^2
     * 
     * Similarly for Eta (x-component) and Zeta (y-component):
     *   Eta.v_x = 0.05*i^2  =>  ∂²η_x/∂x² = 0.1 * DX_INVERSE^2
     *   Zeta.v_y = 0.05*j^2 =>  ∂²ζ_y/∂y² = 0.1 * DY_INVERSE^2
     *   (other components set to have zero second derivatives)
     */
    
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, false);
    
    // Coefficients for quadratic velocity distributions
    const DTYPE coeff_u_x = 0.1;
    const DTYPE coeff_u_y = 0.2;
    const DTYPE coeff_u_z = 0.3;
    const DTYPE coeff_eta_x = 0.05;
    const DTYPE coeff_zeta_y = 0.05;
    
    // Populate all grid points with quadratic velocity fields and linear pressure
    for(int kk = 0; kk < DEPTH; kk++) {
        for(int jj = 0; jj < HEIGHT; jj++) {
            for(int ii = 0; ii < WIDTH; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                // Force field: constant
                f_field.f_x[idx] = 10.0;
                f_field.f_y[idx] = 20.0;
                f_field.f_z[idx] = 30.0;
                
                // Pressure: linear => constant gradients
                // p = 100 + 5*i + 6*j + 7*k
                pressure.p[idx] = 100.0 + 5.0*ii + 6.0*jj + 7.0*kk;
                
                // Permeability constant
                K[idx] = 2.0;
                
                // Velocity U: quadratic to create non-zero second derivatives
                U.v_x[idx] = coeff_u_x * ii * ii;
                U.v_y[idx] = coeff_u_y * jj * jj;
                U.v_z[idx] = coeff_u_z * kk * kk;
                
                // Eta field: quadratic in x for Eta.v_x, others constant
                Eta.v_x[idx] = coeff_eta_x * ii * ii;
                Eta.v_y[idx] = 0.0;  // constant => zero second derivative
                Eta.v_z[idx] = 0.0;
                
                // Zeta field: quadratic in y for Zeta.v_y, others constant
                Zeta.v_x[idx] = 0.0;
                Zeta.v_y[idx] = coeff_zeta_y * jj * jj;
                Zeta.v_z[idx] = 0.0;
            }
        }
    }
    
    // Run compute_g
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
    
    // Verify interior points
    int points_checked = 0;
    
    for(int kk = 1; kk < DEPTH-1; kk++) {
        for(int jj = 1; jj < HEIGHT-1; jj++) {
            for(int ii = 1; ii < WIDTH-1; ii++) {
                size_t idx = rowmaj_idx(ii, jj, kk);
                
                // ===== PRESSURE GRADIENTS =====
                // p = 100 + 5*i + 6*j + 7*k
                // grad_x = 5/DX, grad_y = 6/DY, grad_z = 7/DZ
                DTYPE grad_p_x = 5.0 * DX_INVERSE;
                DTYPE grad_p_y = 6.0 * DY_INVERSE;
                DTYPE grad_p_z = 7.0 * DZ_INVERSE;
                
                // ===== BRINKMAN TERMS =====
                // -(NU / (2*K)) * U
                DTYPE brinkman_x = -(NU / (2.0 * K[idx])) * U.v_x[idx];
                DTYPE brinkman_y = -(NU / (2.0 * K[idx])) * U.v_y[idx];
                DTYPE brinkman_z = -(NU / (2.0 * K[idx])) * U.v_z[idx];
                
                // ===== VISCOUS DIFFUSION TERMS =====
                // For quadratic fields f(i) = c*i^2, the discrete second derivative is:
                // d²f/di² ≈ (f[i-1] - 2*f[i] + f[i+1]) / (Δi)²
                // For f(i) = c*i^2:
                //   f[i-1] = c*(i-1)^2 = c*(i² - 2i + 1)
                //   f[i]   = c*i²
                //   f[i+1] = c*(i+1)^2 = c*(i² + 2i + 1)
                // => f[i-1] - 2*f[i] + f[i+1] = c*(i²-2i+1 - 2i² + i²+2i+1) = 2c
                // => d²f/di² = 2c / (Δi)²
                
                // Compute expected second derivatives analytically
                // ∂²(Eta.v_x)/∂x² where Eta.v_x = coeff_eta_x * i²
                DTYPE d2_eta_x_dx2 = 2.0 * coeff_eta_x * DX_INVERSE_SQUARE;
                
                // ∂²(Zeta.v_y)/∂y² where Zeta.v_y = coeff_zeta_y * j²
                DTYPE d2_zeta_y_dy2 = 2.0 * coeff_zeta_y * DY_INVERSE_SQUARE;

                // ∂²(U.v_z)/∂z² where U.v_z = coeff_u_z * k²
                DTYPE d2_u_z_dz2 = 2.0 * coeff_u_z * DZ_INVERSE_SQUARE;

                // For g_x: viscous term = (NU/2) * [∂²η_x/∂x² + ∂²ζ_x/∂y² + ∂²u_x/∂z²]
                // Since Zeta.v_x and U.v_x (in their respective derivative directions for cross-terms) 
                // are designed to be zero or constant in those directions, we focus on the main terms.
                // Actually, looking at the formula:
                //   g_x = f_x - ∂p/∂x - (ν/2k)u_x + (ν/2)(∂²η_x/∂x² + ∂²ζ_x/∂y² + ∂²u_x/∂z²)
                
                // For our setup:
                //   ∂²(Eta.v_x)/∂x² = d2_eta_x_dx2 (computed above)
                //   ∂²(Zeta.v_x)/∂y² = 0 (Zeta.v_x is constant = 0)
                //   ∂²(U.v_x)/∂z² = 0 (U.v_x = coeff_u_x*i², independent of z)
                DTYPE viscous_x = (NU / 2.0) * (d2_eta_x_dx2 + 0.0 + 0.0);
                
                // For g_y:
                //   ∂²(Eta.v_y)/∂x² = 0 (Eta.v_y is constant = 0)
                //   ∂²(Zeta.v_y)/∂y² = d2_zeta_y_dy2 (computed above)
                //   ∂²(U.v_y)/∂z² = 0 (U.v_y = coeff_u_y*j², independent of z)
                DTYPE viscous_y = (NU / 2.0) * (0.0 + d2_zeta_y_dy2 + 0.0);
                
                // For g_z:
                //   ∂²(Eta.v_z)/∂x² = 0 (Eta.v_z is constant = 0)
                //   ∂²(Zeta.v_z)/∂y² = 0 (Zeta.v_z is constant = 0)
                //   ∂²(U.v_z)/∂z² = d2_u_z_dz2 (computed above)
                DTYPE viscous_z = (NU / 2.0) * (0.0 + 0.0 + d2_u_z_dz2);
                
                // ===== EXPECTED G VALUES =====
                DTYPE expected_g_x = f_field.f_x[idx] - grad_p_x + brinkman_x + viscous_x;
                DTYPE expected_g_y = f_field.f_y[idx] - grad_p_y + brinkman_y + viscous_y;
                DTYPE expected_g_z = f_field.f_z[idx] - grad_p_z + brinkman_z + viscous_z;
                
                // Verify each component with appropriate tolerance
                // Since we're using discrete second derivatives and large values, use relaxed tolerance
                EXPECT_NEAR(g_field.g_x[idx], expected_g_x, 1e-6)
                    << "G_x mismatch at (" << ii << "," << jj << "," << kk << ")"
                    << "\n  Computed: " << g_field.g_x[idx]
                    << "\n  Expected: " << expected_g_x
                    << "\n  f_x: " << f_field.f_x[idx]
                    << "\n  grad_p_x: " << grad_p_x
                    << "\n  brinkman_x: " << brinkman_x
                    << "\n  viscous_x: " << viscous_x;
                
                EXPECT_NEAR(g_field.g_y[idx], expected_g_y, 1e-6)
                    << "G_y mismatch at (" << ii << "," << jj << "," << kk << ")"
                    << "\n  Computed: " << g_field.g_y[idx]
                    << "\n  Expected: " << expected_g_y
                    << "\n  f_y: " << f_field.f_y[idx]
                    << "\n  grad_p_y: " << grad_p_y
                    << "\n  brinkman_y: " << brinkman_y
                    << "\n  viscous_y: " << viscous_y;
                
                EXPECT_NEAR(g_field.g_z[idx], expected_g_z, 1e-6)
                    << "G_z mismatch at (" << ii << "," << jj << "," << kk << ")"
                    << "\n  Computed: " << g_field.g_z[idx]
                    << "\n  Expected: " << expected_g_z
                    << "\n  f_z: " << f_field.f_z[idx]
                    << "\n  grad_p_z: " << grad_p_z
                    << "\n  brinkman_z: " << brinkman_z
                    << "\n  viscous_z: " << viscous_z;
                
                points_checked++;
            }
        }
    }
    
    // Verify point count
    EXPECT_EQ(points_checked, (WIDTH-2) * (HEIGHT-2) * (DEPTH-2))
        << "Should check all interior points";
    
    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);
}
