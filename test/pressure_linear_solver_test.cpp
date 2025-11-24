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
#include "pressure_system.h"
#include "tridiagonal_blocks.h"
#include "pressure.h"
}

void setup_test_fields(VelocityField *U_next, 
                           Pressure *psi, 
                           Pressure *phi_lower, 
                           Pressure *phi_higher,
                           Pressure *pressure) {


    U_next->v_x = (DTYPE*) malloc(GRID_SIZE);
    U_next->v_y = (DTYPE*) malloc(GRID_SIZE);
    U_next->v_z = (DTYPE*) malloc(GRID_SIZE);
    psi->p = (DTYPE*) malloc(GRID_SIZE);
    phi_lower->p = (DTYPE*) malloc(GRID_SIZE);
    phi_higher->p = (DTYPE*) malloc(GRID_SIZE);
    pressure->p = (DTYPE*) malloc(GRID_SIZE);
    
    rand_fill(pressure->p);
    rand_fill(U_next->v_x);
    rand_fill(U_next->v_y);
    rand_fill(U_next->v_z);
    memset(psi->p, 0, GRID_SIZE);
    memset(phi_lower->p, 0, GRID_SIZE);
    memset(phi_higher->p, 0, GRID_SIZE);
}

void cleanup_test_fields(VelocityField *U_next, 
                           Pressure *psi, 
                           Pressure *phi_lower, 
                           Pressure *phi_higher,
                           Pressure *pressure) {
    free(U_next->v_x);
    free(U_next->v_y);
    free(U_next->v_z);
    free(psi->p);
    free(phi_lower->p);
    free(phi_higher->p);
    free(pressure->p);
}

TEST(Pressure_linear_solver_test, Psi_computation) {
    VelocityField U_next; 
    Pressure psi;
    Pressure phi_lower; 
    Pressure phi_higher;
    Pressure pressure;
    
    setup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);

    solve_pressure_system(U_next, &pressure);


    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = WIDTH;  // Size of a single row (one tridiagonal system)
    
    double w = - DX_INVERSE_SQUARE;

    Pressure rhs_matrix;
    initialize_pressure(&rhs_matrix);
     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs_matrix.p[idx] = (compute_velocity_x_grad(U_next.v_x, i, j, k) +
                              compute_velocity_y_grad(U_next.v_y, i, j, k) +
                              compute_velocity_z_grad(U_next.v_z, i, j, k)) *  (-1.0 /DT);
            }
        }
    }
    
    // Build matrix A for a single row where A = (I - γ∂²/∂x²)
    // From slide: diagonal = (1 + 2γΔx⁻²), off-diagonals = -γΔx⁻²
    // With w = -γΔx⁻², this becomes:
    // Diagonal: 1 - 2*w
    // Off-diagonals: w (symmetric)

    SpMat mat(n, n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(3 * n - 2);
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
            if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
            if (i + 1 < n) triplets.emplace_back(i, i + 1, w);  // super-diagonal
        }
        mat.setFromTriplets(triplets.begin(), triplets.end());


    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            // Build RHS for this specific row (j, k)
            SpVec rhs_vector(n);
            for (int i = 0; i < n; ++i) {
                size_t idx = rowmaj_idx(i, j, k);
                rhs_vector[i] = rhs_matrix.p[idx];
            }

            // Solve with GMRES
            double tol = 1.e-13;
            int maxit = 1000;

            Eigen::GMRES<SpMat> gmres;
            gmres.setMaxIterations(maxit);
            gmres.setTolerance(tol);
            gmres.compute(mat);
            SpVec s = gmres.solve(rhs_vector);

            // Compare C solver output (psi) with Eigen GMRES solution for this row
            for (int i = 0; i < WIDTH; ++i) {
                size_t idx = rowmaj_idx(i, j, k);
                double expected = s[i];
                EXPECT_NEAR(psi.p[idx], expected, 1e-10) 
                    << "Row (j=" << j << ",k=" << k << "), i=" << i << " (idx=" << idx << ")"
                    << ": C solver psi = " << psi.p[idx] << ", Eigen = " << expected;
            }
        }
    }

    free_pressure(&rhs_matrix);
    cleanup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);
}

TEST(Pressure_linear_solver_test, Phi_lower_computation) {
    VelocityField U_next; 
    Pressure psi;
    Pressure phi_lower; 
    Pressure phi_higher;
    Pressure pressure;
    
    setup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);

    solve_pressure_system(U_next, &pressure);


    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = HEIGHT;
    
    double w = - DY_INVERSE_SQUARE;

    Pressure rhs_matrix;
    initialize_pressure(&rhs_matrix);
     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs_matrix.p[idx] = psi.p[idx];
            }
        }
    }

    SpMat mat(n, n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(3 * n - 2);
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
            if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
            if (i + 1 < n) triplets.emplace_back(i, i + 1, w);  // super-diagonal
        }
        mat.setFromTriplets(triplets.begin(), triplets.end());


    for (int k = 0; k < DEPTH; k++) {
        for (int i = 0; i < WIDTH; i++) {
            // Build RHS for this specific row (j, k)
            SpVec rhs_vector(n);
            for (int j = 0; j < n; ++j) {
                size_t idx = rowmaj_idx(i, j, k);
                rhs_vector[j] = rhs_matrix.p[idx];
            }

            // Solve with GMRES
            double tol = 1.e-13;
            int maxit = 1000;

            Eigen::GMRES<SpMat> gmres;
            gmres.setMaxIterations(maxit);
            gmres.setTolerance(tol);
            gmres.compute(mat);
            SpVec s = gmres.solve(rhs_vector);

            // Compare C solver output (psi) with Eigen GMRES solution for this row
            for (int j = 0; j < HEIGHT; ++j) {
                size_t idx = rowmaj_idx(i, j, k);
                double expected = s[j];
                EXPECT_NEAR(phi_lower.p[idx], expected, 1e-10) 
                    << "Row (j=" << j << ",k=" << k << "), i=" << i << " (idx=" << idx << ")"
                    << ": C solver phi_lower = " << phi_lower.p[idx] << ", Eigen = " << expected;
            }
        }
    }

    free_pressure(&rhs_matrix);
    cleanup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);
}

TEST(Pressure_linear_solver_test, Phi_higher_computation) {
    VelocityField U_next; 
    Pressure psi;
    Pressure phi_lower; 
    Pressure phi_higher;
    Pressure pressure;
    
    setup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);

    solve_pressure_system(U_next, &pressure);


    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = DEPTH;
    
    double w = - DZ_INVERSE_SQUARE;

    Pressure rhs_matrix;
    initialize_pressure(&rhs_matrix);
     for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);

                rhs_matrix.p[idx] = phi_lower.p[idx];
            }
        }
    }

    SpMat mat(n, n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(3 * n - 2);
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 1.0 - 2.0 * w);  // main diagonal
            if (i > 0) triplets.emplace_back(i, i - 1, w);  // sub-diagonal
            if (i + 1 < n) triplets.emplace_back(i, i + 1, w);  // super-diagonal
        }
        mat.setFromTriplets(triplets.begin(), triplets.end());


    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            // Build RHS for this specific row (j, k)
            SpVec rhs_vector(n);
            for (int k = 0; k < n; ++k) {
                size_t idx = rowmaj_idx(i, j, k);
                rhs_vector[k] = rhs_matrix.p[idx];
            }

            // Solve with GMRES
            double tol = 1.e-13;
            int maxit = 1000;

            Eigen::GMRES<SpMat> gmres;
            gmres.setMaxIterations(maxit);
            gmres.setTolerance(tol);
            gmres.compute(mat);
            SpVec s = gmres.solve(rhs_vector);

            // Compare C solver output (psi) with Eigen GMRES solution for this row
            for (int k = 0; k < DEPTH; ++k) {
                size_t idx = rowmaj_idx(i, j, k);
                double expected = s[k];
                EXPECT_NEAR(phi_higher.p[idx], expected, 1e-10) 
                    << "Row (j=" << j << ",k=" << k << "), i=" << i << " (idx=" << idx << ")"
                    << ": C solver phi_higher = " << phi_higher.p[idx] << ", Eigen = " << expected;
            }
        }
    }

    free_pressure(&rhs_matrix);
    cleanup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);
}