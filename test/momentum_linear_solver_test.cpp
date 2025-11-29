#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
// GMRES solver is in the unsupported IterativeSolvers module
#include <unsupported/Eigen/IterativeSolvers>

// Include C headers in extern "C" block for C++ compatibility
extern "C" {
#include "g_field.h"
#include "force_field.h"
#include "velocity_field.h"
#include "pressure.h"
#include "momentum_system.h"
}

void setup_test_fields(GField* g_field, ForceField* f_field, Pressure* pressure,
                       DTYPE** K, VelocityField* Eta, VelocityField* Zeta, 
                       VelocityField* U, VelocityField* Eta_next, 
                       VelocityField* Zeta_next, VelocityField* U_next, 
                       VelocityField* Xi, bool initialize_to_zero = true) {
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
    Eta_next->v_x = (DTYPE*) malloc(GRID_SIZE);
    Eta_next->v_y = (DTYPE*) malloc(GRID_SIZE);
    Eta_next->v_z = (DTYPE*) malloc(GRID_SIZE);
    Zeta_next->v_x = (DTYPE*) malloc(GRID_SIZE);
    Zeta_next->v_y = (DTYPE*) malloc(GRID_SIZE);
    Zeta_next->v_z = (DTYPE*) malloc(GRID_SIZE);
    U_next->v_x = (DTYPE*) malloc(GRID_SIZE);
    U_next->v_y = (DTYPE*) malloc(GRID_SIZE);
    U_next->v_z = (DTYPE*) malloc(GRID_SIZE);
    Xi->v_x = (DTYPE*) malloc(GRID_SIZE);
    Xi->v_y = (DTYPE*) malloc(GRID_SIZE);
    Xi->v_z = (DTYPE*) malloc(GRID_SIZE);
    
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
        memset(Eta_next->v_x, 0, GRID_SIZE);
        memset(Eta_next->v_y, 0, GRID_SIZE);
        memset(Eta_next->v_z, 0, GRID_SIZE);
        memset(Zeta_next->v_x, 0, GRID_SIZE);
        memset(Zeta_next->v_y, 0, GRID_SIZE);
        memset(Zeta_next->v_z, 0, GRID_SIZE);
        memset(U_next->v_x, 0, GRID_SIZE);
        memset(U_next->v_y, 0, GRID_SIZE);
        memset(U_next->v_z, 0, GRID_SIZE);
        memset(Xi->v_x, 0, GRID_SIZE);
        memset(Xi->v_y, 0, GRID_SIZE);
        memset(Xi->v_z, 0, GRID_SIZE);
    }
}

// Helper to cleanup all fields
void cleanup_test_fields(GField* g_field, ForceField* f_field, Pressure* pressure,
                        DTYPE* K, VelocityField* Eta, VelocityField* Zeta, 
                        VelocityField* U, VelocityField* Eta_next, 
                        VelocityField* Zeta_next, VelocityField* U_next, 
                        VelocityField* Xi) {
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
    /* free next/temporary fields allocated in setup (they are pointer arguments) */
    free(Eta_next->v_x);
    free(Eta_next->v_y);
    free(Eta_next->v_z);
    free(Zeta_next->v_x);
    free(Zeta_next->v_y);
    free(Zeta_next->v_z);
    free(U_next->v_x);
    free(U_next->v_y);
    free(U_next->v_z);
    free(Xi->v_x);
    free(Xi->v_y);
    free(Xi->v_z);
}

TEST(Momentum_linear_solver_test, Xi_computation) {
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    VelocityField Eta_next, Zeta_next, U_next, Xi;
    DTYPE *u_BC_current_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_second_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_third_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);

    /* allocate Beta and Gamma (bytes) */
    DTYPE *Beta = (DTYPE*) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE*) malloc(GRID_SIZE);
    ASSERT_NE(Gamma, nullptr);
    ASSERT_NE(Beta, nullptr);

    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        Beta[idx] = 0.005;
        Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
    }

    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        U.v_x[idx] = 2.0;
        U.v_y[idx] = 3.0;
        U.v_z[idx] = 4.0;
        g_field.g_x[idx] = 7.0;
        g_field.g_y[idx] = 5.0;
        g_field.g_z[idx] = 3.0;
    }

    solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma,
        u_BC_current_direction,
        u_BC_derivative_second_direction,
        u_BC_derivative_third_direction);


    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        EXPECT_NEAR(Xi.v_x[idx], 3.4, 1e-12);
        EXPECT_NEAR(Xi.v_y[idx], 4.0, 1e-12);
        EXPECT_NEAR(Xi.v_z[idx], 4.6, 1e-12);
    }

    free(Beta);
    free(Gamma);

    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);
}

TEST(Momentum_linear_solver_test, Eta_next_computation) {
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    VelocityField Eta_next, Zeta_next, U_next, Xi;
    DTYPE *u_BC_current_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_second_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_third_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);

    /* allocate Beta and Gamma (bytes) */
    DTYPE *Beta = (DTYPE*) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE*) malloc(GRID_SIZE);
    ASSERT_NE(Gamma, nullptr);
    ASSERT_NE(Beta, nullptr);

    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        Beta[idx] = 0.005;
        Gamma[idx] = 0.000005;
        Eta.v_x[idx] = 1.5;
        Eta.v_y[idx] = 7.7;
        Eta.v_z[idx] = 5.8;
        U.v_x[idx] = 2.0;
        U.v_y[idx] = 3.0;
        U.v_z[idx] = 4.0;
        g_field.g_x[idx] = 7.0;
        g_field.g_y[idx] = 5.0;
        g_field.g_z[idx] = 3.0;
    }

    // Calculate Xi first (Xi = U + (dt/Beta) * g)
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                
                DTYPE coeff = DT / Beta[idx];

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];
                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];
                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];
            }
        }
    }

    // Set boundary conditions to zero
    // This should allow the solver to determine boundary values naturally
    memset(u_BC_current_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_second_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_third_direction, 0, GRID_SIZE * sizeof(DTYPE));

    solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma,
        u_BC_current_direction,
        u_BC_derivative_second_direction,
        u_BC_derivative_third_direction);


    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        EXPECT_NEAR(Xi.v_x[idx], 3.4, 1e-12);
        EXPECT_NEAR(Xi.v_y[idx], 4.0, 1e-12);
        EXPECT_NEAR(Xi.v_z[idx], 4.6, 1e-12);
    }

    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = WIDTH;  // Size of a single row (one tridiagonal system)
    
    // Compute w coefficient: w = -Gamma * DX_INVERSE_SQUARE
    // (assuming Gamma is constant across grid for this test)
    double w = -Gamma[0] * DX_INVERSE_SQUARE;
    
    // Build matrix A for a single row where A = (I - γ∂²/∂x²)
    // With Dirichlet boundary conditions: s[0] = 0 and s[n-1] = 0
    // This means Eta_next[0] = Eta[0] and Eta_next[n-1] = Eta[n-1]
    // We solve for interior points only (indices 1 to n-2)
    const int n_interior = n - 2;
    SpMat mat(n_interior, n_interior);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * n_interior);
    
    for (int i = 0; i < n_interior; ++i) {
        triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
        if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
        if (i + 1 < n_interior) triplets.emplace_back(i, i + 1, w);  // super-diagonal
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());

    // Build RHS for interior points only
    SpVec rhs(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(i + 1, 0, 0);  // Interior points: i=1 to i=n-2
        rhs[i] = Xi.v_x[idx] - Eta.v_x[idx];
    }
    // Adjust for boundary conditions (s[0] = 0 and s[n-1] = 0)
    // First interior point (i=1): affected by s[0] = 0
    // rhs[0] -= w * 0  (no change)
    // Last interior point (i=n-2): affected by s[n-1] = 0
    // rhs[n_interior-1] -= w * 0  (no change)

    // Check matrix properties
    std::cout << "Testing first row (k=0, j=0) with Dirichlet BCs:" << std::endl;
    std::cout << "Matrix size (interior points): " << mat.rows() << "x" << mat.cols() << std::endl;
    std::cout << "Non zero entries: " << mat.nonZeros() << std::endl;
    std::cout << "w coefficient: " << w << std::endl;
    SpMat B = SpMat(mat.transpose()) - mat;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

    // Set parameters for solver
    double tol = 1.e-13;                 // Convergence tolerance
    int maxit = 1000;                     // Maximum iterations

    // Solve with GMRES: mat * s_interior = rhs, where s_interior = s[1:n-2]
    Eigen::GMRES<SpMat> gmres;
    gmres.setMaxIterations(maxit);
    gmres.setTolerance(tol);
    gmres.compute(mat);
    SpVec s_interior = gmres.solve(rhs);  // s_interior for interior points
    
    std::cout << "Eigen GMRES solver:" << std::endl;
    std::cout << "#iterations:     " << gmres.iterations() << std::endl;
    std::cout << "relative residual: " << gmres.error() << std::endl;
    
    // Compute residual ||As - rhs|| as a measure of solution quality
    double residual_norm = (mat * s_interior - rhs).norm();
    std::cout << "residual norm ||As-rhs||: " << residual_norm << std::endl;

    // Compare: Eta_next from C solver should match
    // Boundary points: Eta_next[0] = Eta[0], Eta_next[n-1] = Eta[n-1]
    // Interior points: Eta_next[i] = s_interior[i-1] + Eta[i]
    std::cout << "Comparing C solver (Eta_next.v_x) vs Eigen GMRES solution for first row:" << std::endl;
    
    // Check boundary points
    size_t idx0 = rowmaj_idx(0, 0, 0);
    size_t idx_last = rowmaj_idx(WIDTH - 1, 0, 0);
    std::cout << "Boundary at i=0: C solver = " << Eta_next.v_x[idx0] 
              << ", Expected (Eta) = " << Eta.v_x[idx0] << std::endl;
    std::cout << "Boundary at i=" << (WIDTH-1) << ": C solver = " << Eta_next.v_x[idx_last]
              << ", Expected (Eta) = " << Eta.v_x[idx_last] << std::endl;
    
    // Check interior points
    for (int i = 1; i < WIDTH - 1; ++i) {
        size_t idx = rowmaj_idx(i, 0, 0);
        double expected = s_interior[i - 1] + Eta.v_x[idx];  // Eta_next = s + Eta
        EXPECT_NEAR(Eta_next.v_x[idx], expected, 1e-10) 
            << "Mismatch at index " << i 
            << " (grid index " << idx << ")"
            << ": C solver = " << Eta_next.v_x[idx] 
            << ", Eigen = " << expected;
    }

    // Test v_y component: also solved along x-direction (WIDTH) using same matrix
    // compute_eta_next calls solve_Dxx_tridiag_blocks for ALL components (v_x, v_y, v_z)
    SpVec rhs_vy(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(i + 1, 0, 0);  // Interior points: i=1 to i=WIDTH-2
        rhs_vy[i] = Xi.v_y[idx] - Eta.v_y[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_y
    Eigen::GMRES<SpMat> gmres_vy;
    gmres_vy.setMaxIterations(maxit);
    gmres_vy.setTolerance(tol);
    gmres_vy.compute(mat);  // Same matrix as v_x (interior-only, DX_INVERSE_SQUARE)
    SpVec s_vy_interior = gmres_vy.solve(rhs_vy);

    double residual_norm_vy = (mat * s_vy_interior - rhs_vy).norm();
    std::cout << "residual norm (v_y) ||As-rhs||: " << residual_norm_vy << std::endl;

    for (int i = 1; i < WIDTH - 1; ++i) {
        size_t idx = rowmaj_idx(i, 0, 0);
        double expected = s_vy_interior[i - 1] + Eta.v_y[idx];
        EXPECT_NEAR(Eta_next.v_y[idx], expected, 1e-10)
            << "Mismatch at v_y index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << Eta_next.v_y[idx]
            << ", Eigen = " << expected;
    }

    SpVec rhs_vz(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(i + 1, 0, 0);  // Interior points: i=1 to i=WIDTH-2
        rhs_vz[i] = Xi.v_z[idx] - Eta.v_z[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_z
    Eigen::GMRES<SpMat> gmres_vz;
    gmres_vz.setMaxIterations(maxit);
    gmres_vz.setTolerance(tol);
    gmres_vz.compute(mat);  // Same matrix as v_x (interior-only, DX_INVERSE_SQUARE)
    SpVec s_vz_interior = gmres_vz.solve(rhs_vz);
    double residual_norm_vz = (mat * s_vz_interior - rhs_vz).norm();
    std::cout << "residual norm (v_z) ||As-rhs||: " << residual_norm_vz << std::endl;

    for (int i = 1; i < WIDTH - 1; ++i) {
        size_t idx = rowmaj_idx(i, 0, 0);
        double expected = s_vz_interior[i - 1] + Eta.v_z[idx];
        EXPECT_NEAR(Eta_next.v_z[idx], expected, 1e-10)
            << "Mismatch at v_z index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << Eta_next.v_z[idx]
            << ", Eigen = " << expected;
    }


    free(Beta);
    free(Gamma);

    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);
}

TEST(Momentum_linear_solver_test, Zeta_next_computation) {
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    VelocityField Eta_next, Zeta_next, U_next, Xi;
    DTYPE *u_BC_current_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_second_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_third_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);

    /* allocate Beta and Gamma (bytes) */
    DTYPE *Beta = (DTYPE*) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE*) malloc(GRID_SIZE);
    ASSERT_NE(Gamma, nullptr);
    ASSERT_NE(Beta, nullptr);

    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        Beta[idx] = 0.005;
        Gamma[idx] = 0.000005;
        Eta.v_x[idx] = 1.5;
        Eta.v_y[idx] = 7.7;
        Eta.v_z[idx] = 5.8;
        U.v_x[idx] = 2.0;
        U.v_y[idx] = 3.0;
        U.v_z[idx] = 4.0;
        g_field.g_x[idx] = 7.0;
        g_field.g_y[idx] = 5.0;
        g_field.g_z[idx] = 3.0;
    }

    // Calculate Xi first (Xi = U + (dt/Beta) * g)
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                
                DTYPE coeff = DT / Beta[idx];

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];
                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];
                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];
            }
        }
    }

    // Set boundary conditions to zero
    memset(u_BC_current_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_second_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_third_direction, 0, GRID_SIZE * sizeof(DTYPE));

    solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma,
        u_BC_current_direction,
        u_BC_derivative_second_direction,
        u_BC_derivative_third_direction);


    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        EXPECT_NEAR(Xi.v_x[idx], 3.4, 1e-12);
        EXPECT_NEAR(Xi.v_y[idx], 4.0, 1e-12);
        EXPECT_NEAR(Xi.v_z[idx], 4.6, 1e-12);
    }

    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = HEIGHT;
    double w = -Gamma[0] * DY_INVERSE_SQUARE;

    // Build matrix for interior points only (Dirichlet BC)
    const int n_interior = n - 2;
    SpMat mat(n_interior, n_interior);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * n_interior);
    
    for (int i = 0; i < n_interior; ++i) {
        triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
        if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
        if (i + 1 < n_interior) triplets.emplace_back(i, i + 1, w);  // super-diagonal
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());

    // Build RHS for interior points only
    SpVec rhs(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, i + 1, 0);  // Interior points: j=1 to j=HEIGHT-2
        rhs[i] = Eta_next.v_x[idx] - Zeta.v_x[idx];
    }

    // Check matrix properties
    std::cout << "Testing first column (i=0, k=0) with Dirichlet BCs:" << std::endl;
    std::cout << "Matrix size (interior points): " << mat.rows() << "x" << mat.cols() << std::endl;
    std::cout << "Non zero entries: " << mat.nonZeros() << std::endl;
    std::cout << "w coefficient: " << w << std::endl;
    SpMat B = SpMat(mat.transpose()) - mat;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

    // Set parameters for solver
    double tol = 1.e-13;                 // Convergence tolerance
    int maxit = 1000;                     // Maximum iterations

    // Solve with GMRES: mat * s_interior = rhs
    Eigen::GMRES<SpMat> gmres;
    gmres.setMaxIterations(maxit);
    gmres.setTolerance(tol);
    gmres.compute(mat);
    SpVec s_interior = gmres.solve(rhs);  // s_interior for interior points
    
    std::cout << "Eigen GMRES solver:" << std::endl;
    std::cout << "#iterations:     " << gmres.iterations() << std::endl;
    std::cout << "relative residual: " << gmres.error() << std::endl;
    
    // Compute residual ||As - rhs|| as a measure of solution quality
    double residual_norm = (mat * s_interior - rhs).norm();
    std::cout << "residual norm ||As-rhs||: " << residual_norm << std::endl;

    // Compare: Zeta_next from C solver should match
    // Boundary points: Zeta_next[j=0] = Zeta[j=0], Zeta_next[j=HEIGHT-1] = Zeta[j=HEIGHT-1]
    // Interior points: Zeta_next[j] = s_interior[j-1] + Zeta[j]
    std::cout << "Comparing C solver (Zeta_next.v_x) vs Eigen GMRES solution:" << std::endl;
    
    // Check boundary points
    size_t idx0 = rowmaj_idx(0, 0, 0);
    size_t idx_last = rowmaj_idx(0, HEIGHT - 1, 0);
    std::cout << "Boundary at j=0: C solver = " << Zeta_next.v_x[idx0] 
              << ", Expected (Zeta) = " << Zeta.v_x[idx0] << std::endl;
    std::cout << "Boundary at j=" << (HEIGHT-1) << ": C solver = " << Zeta_next.v_x[idx_last]
              << ", Expected (Zeta) = " << Zeta.v_x[idx_last] << std::endl;
    
    // Check interior points
    for (int i = 1; i < HEIGHT - 1; ++i) {
        size_t idx = rowmaj_idx(0, i, 0);
        double expected = s_interior[i - 1] + Zeta.v_x[idx];
        EXPECT_NEAR(Zeta_next.v_x[idx], expected, 1e-10) 
            << "Mismatch at index " << i 
            << " (grid index " << idx << ")"
            << ": C solver = " << Zeta_next.v_x[idx] 
            << ", Eigen = " << expected;
    }

    SpVec rhs_vy(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, i + 1, 0);  // Interior points
        rhs_vy[i] = Eta_next.v_y[idx] - Zeta.v_y[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_y
    Eigen::GMRES<SpMat> gmres_vy;
    gmres_vy.setMaxIterations(maxit);
    gmres_vy.setTolerance(tol);
    gmres_vy.compute(mat);
    SpVec s_vy_interior = gmres_vy.solve(rhs_vy);

    double residual_norm_vy = (mat * s_vy_interior - rhs_vy).norm();
    std::cout << "residual norm (v_y) ||As-rhs||: " << residual_norm_vy << std::endl;

    for (int i = 1; i < HEIGHT - 1; ++i) {
        size_t idx = rowmaj_idx(0, i, 0);
        double expected = s_vy_interior[i - 1] + Zeta.v_y[idx];
        EXPECT_NEAR(Zeta_next.v_y[idx], expected, 1e-10)
            << "Mismatch at v_y index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << Zeta_next.v_y[idx]
            << ", Eigen = " << expected;
    }

    SpVec rhs_vz(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, i + 1, 0);  // Interior points
        rhs_vz[i] = Eta_next.v_z[idx] - Zeta.v_z[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_z
    Eigen::GMRES<SpMat> gmres_vz;
    gmres_vz.setMaxIterations(maxit);
    gmres_vz.setTolerance(tol);
    gmres_vz.compute(mat);
    SpVec s_vz_interior = gmres_vz.solve(rhs_vz);
    double residual_norm_vz = (mat * s_vz_interior - rhs_vz).norm();
    std::cout << "residual norm (v_z) ||As-rhs||: " << residual_norm_vz << std::endl;

    for (int i = 1; i < HEIGHT - 1; ++i) {
        size_t idx = rowmaj_idx(0, i, 0);
        double expected = s_vz_interior[i - 1] + Zeta.v_z[idx];
        EXPECT_NEAR(Zeta_next.v_z[idx], expected, 1e-10)
            << "Mismatch at v_z index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << Zeta_next.v_z[idx]
            << ", Eigen = " << expected;
    }


    free(Beta);
    free(Gamma);

    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);
}

TEST(Momentum_linear_solver_test, U_next_computation) {
    GField g_field;
    ForceField f_field;
    Pressure pressure;
    DTYPE* K;
    VelocityField Eta, Zeta, U;
    VelocityField Eta_next, Zeta_next, U_next, Xi;
    DTYPE *u_BC_current_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_second_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    DTYPE *u_BC_derivative_third_direction = (DTYPE*) malloc(sizeof(DTYPE) * GRID_SIZE);
    
    setup_test_fields(&g_field, &f_field, &pressure, &K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);

    /* allocate Beta and Gamma (bytes) */
    DTYPE *Beta = (DTYPE*) malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE*) malloc(GRID_SIZE);
    ASSERT_NE(Gamma, nullptr);
    ASSERT_NE(Beta, nullptr);

    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        Beta[idx] = 0.005;
        Gamma[idx] = 0.000005;
        Eta.v_x[idx] = 1.5;
        Eta.v_y[idx] = 7.7;
        Eta.v_z[idx] = 5.8;
        U.v_x[idx] = 2.0;
        U.v_y[idx] = 3.0;
        U.v_z[idx] = 4.0;
        g_field.g_x[idx] = 7.0;
        g_field.g_y[idx] = 5.0;
        g_field.g_z[idx] = 3.0;
    }

    // Calculate Xi first (Xi = U + (dt/Beta) * g)
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                
                DTYPE coeff = DT / Beta[idx];

                Xi.v_x[idx] = U.v_x[idx] + coeff * g_field.g_x[idx];
                Xi.v_y[idx] = U.v_y[idx] + coeff * g_field.g_y[idx];
                Xi.v_z[idx] = U.v_z[idx] + coeff * g_field.g_z[idx];
            }
        }
    }

    // Set boundary conditions to zero
    memset(u_BC_current_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_second_direction, 0, GRID_SIZE * sizeof(DTYPE));
    memset(u_BC_derivative_third_direction, 0, GRID_SIZE * sizeof(DTYPE));

    solve_momentum_system(U, Eta, Zeta, Xi, g_field, U_next, Eta_next, Zeta_next, Beta, Gamma,
        u_BC_current_direction,
        u_BC_derivative_second_direction,
        u_BC_derivative_third_direction);


    for (size_t idx = 0; idx < GRID_ELEMENTS; ++idx) {
        EXPECT_NEAR(Xi.v_x[idx], 3.4, 1e-12);
        EXPECT_NEAR(Xi.v_y[idx], 4.0, 1e-12);
        EXPECT_NEAR(Xi.v_z[idx], 4.6, 1e-12);
    }

    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = DEPTH;
    double w = -Gamma[0] * DZ_INVERSE_SQUARE;

    // Build matrix for interior points only (Dirichlet BC)
    const int n_interior = n - 2;
    SpMat mat(n_interior, n_interior);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * n_interior);
    
    for (int i = 0; i < n_interior; ++i) {
        triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
        if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
        if (i + 1 < n_interior) triplets.emplace_back(i, i + 1, w);  // super-diagonal
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());

    // Build RHS for interior points only
    SpVec rhs(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, 0, i + 1);  // Interior points: k=1 to k=DEPTH-2
        rhs[i] = Zeta_next.v_x[idx] - U.v_x[idx];
    }

    // Check matrix properties
    std::cout << "Testing first line (i=0, j=0) with Dirichlet BCs:" << std::endl;
    std::cout << "Matrix size (interior points): " << mat.rows() << "x" << mat.cols() << std::endl;
    std::cout << "Non zero entries: " << mat.nonZeros() << std::endl;
    std::cout << "w coefficient: " << w << std::endl;
    SpMat B = SpMat(mat.transpose()) - mat;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

    // Set parameters for solver
    double tol = 1.e-13;                 // Convergence tolerance
    int maxit = 1000;                     // Maximum iterations

    // Solve with GMRES: mat * s_interior = rhs
    Eigen::GMRES<SpMat> gmres;
    gmres.setMaxIterations(maxit);
    gmres.setTolerance(tol);
    gmres.compute(mat);
    SpVec s_interior = gmres.solve(rhs);  // s_interior for interior points
    
    std::cout << "Eigen GMRES solver:" << std::endl;
    std::cout << "#iterations:     " << gmres.iterations() << std::endl;
    std::cout << "relative residual: " << gmres.error() << std::endl;
    
    // Compute residual ||As - rhs|| as a measure of solution quality
    double residual_norm = (mat * s_interior - rhs).norm();
    std::cout << "residual norm ||As-rhs||: " << residual_norm << std::endl;

    // Compare: U_next from C solver should match
    // Boundary points: U_next[k=0] = U[k=0], U_next[k=DEPTH-1] = U[k=DEPTH-1]
    // Interior points: U_next[k] = s_interior[k-1] + U[k]
    std::cout << "Comparing C solver (U_next.v_x) vs Eigen GMRES solution:" << std::endl;
    
    // Check boundary points
    size_t idx0 = rowmaj_idx(0, 0, 0);
    size_t idx_last = rowmaj_idx(0, 0, DEPTH - 1);
    std::cout << "Boundary at k=0: C solver = " << U_next.v_x[idx0] 
              << ", Expected (U) = " << U.v_x[idx0] << std::endl;
    std::cout << "Boundary at k=" << (DEPTH-1) << ": C solver = " << U_next.v_x[idx_last]
              << ", Expected (U) = " << U.v_x[idx_last] << std::endl;
    
    // Check interior points
    for (int i = 1; i < DEPTH - 1; ++i) {
        size_t idx = rowmaj_idx(0, 0, i);
        double expected = s_interior[i - 1] + U.v_x[idx];
        EXPECT_NEAR(U_next.v_x[idx], expected, 1e-10) 
            << "Mismatch at index " << i 
            << " (grid index " << idx << ")"
            << ": C solver = " << U_next.v_x[idx] 
            << ", Eigen = " << expected;
    }

    SpVec rhs_vy(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, 0, i + 1);  // Interior points
        rhs_vy[i] = Zeta_next.v_y[idx] - U.v_y[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_y
    Eigen::GMRES<SpMat> gmres_vy;
    gmres_vy.setMaxIterations(maxit);
    gmres_vy.setTolerance(tol);
    gmres_vy.compute(mat);
    SpVec s_vy_interior = gmres_vy.solve(rhs_vy);

    double residual_norm_vy = (mat * s_vy_interior - rhs_vy).norm();
    std::cout << "residual norm (v_y) ||As-rhs||: " << residual_norm_vy << std::endl;

    for (int i = 1; i < DEPTH - 1; ++i) {
        size_t idx = rowmaj_idx(0, 0, i);
        double expected = s_vy_interior[i - 1] + U.v_y[idx];
        EXPECT_NEAR(U_next.v_y[idx], expected, 1e-10)
            << "Mismatch at v_y index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << U_next.v_y[idx]
            << ", Eigen = " << expected;
    }

    SpVec rhs_vz(n_interior);
    for (int i = 0; i < n_interior; ++i) {
        size_t idx = rowmaj_idx(0, 0, i + 1);  // Interior points
        rhs_vz[i] = Zeta_next.v_z[idx] - U.v_z[idx];
    }

    // Reuse same matrix (mat) and solver parameters for v_z
    Eigen::GMRES<SpMat> gmres_vz;
    gmres_vz.setMaxIterations(maxit);
    gmres_vz.setTolerance(tol);
    gmres_vz.compute(mat);
    SpVec s_vz_interior = gmres_vz.solve(rhs_vz);
    double residual_norm_vz = (mat * s_vz_interior - rhs_vz).norm();
    std::cout << "residual norm (v_z) ||As-rhs||: " << residual_norm_vz << std::endl;

    for (int i = 1; i < DEPTH - 1; ++i) {
        size_t idx = rowmaj_idx(0, 0, i);
        double expected = s_vz_interior[i - 1] + U.v_z[idx];
        EXPECT_NEAR(U_next.v_z[idx], expected, 1e-10)
            << "Mismatch at v_z index " << i
            << " (grid index " << idx << ")"
            << ": C solver = " << U_next.v_z[idx]
            << ", Eigen = " << expected;
    }


    free(Beta);
    free(Gamma);

    cleanup_test_fields(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U, &Eta_next, &Zeta_next, &U_next, &Xi);
}