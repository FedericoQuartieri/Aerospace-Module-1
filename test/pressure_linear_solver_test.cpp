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

    solve_pressure_system(U_next, &psi, &phi_lower, &phi_higher, &pressure);


    // Test a single row (block) instead of the entire grid
    // solve_Dxx_tridiag_blocks solves HEIGHT*DEPTH independent tridiagonal systems,
    // each of size WIDTH. We test the first row (k=0, j=0).
    using SpMat = Eigen::SparseMatrix<double>;
    using SpVec = Eigen::VectorXd;

    const int n = WIDTH;  // Size of a single row (one tridiagonal system)
    
    double w = - DX_INVERSE_SQUARE;
    
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

    // Build RHS for the first row (k=0, j=0): b = Xi - Eta for the x-component
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

    SpVec rhs_vector(n);
    for (int i = 0; i < n; ++i) {
        size_t idx = rowmaj_idx(i, 0, 0);  // First row: j=0, k=0
        rhs_vector[i] = rhs_matrix.p[idx];
    }

    // Check matrix properties
    std::cout << "Testing first row (k=0, j=0):" << std::endl;
    std::cout << "Matrix size: " << mat.rows() << "x" << mat.cols() << std::endl;
    std::cout << "Non zero entries: " << mat.nonZeros() << std::endl;
    std::cout << "w coefficient: " << w << std::endl;
    SpMat B = SpMat(mat.transpose()) - mat;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

    // Set parameters for solver
    double tol = 1.e-13;                 // Convergence tolerance
    int maxit = 1000;                     // Maximum iterations

    Eigen::GMRES<SpMat> gmres;
    gmres.setMaxIterations(maxit);
    gmres.setTolerance(tol);
    gmres.compute(mat);
    SpVec s = gmres.solve(rhs_vector); 
    
    std::cout << "Eigen GMRES solver:" << std::endl;
    std::cout << "#iterations:     " << gmres.iterations() << std::endl;
    std::cout << "relative residual: " << gmres.error() << std::endl;
    
    // Compute residual ||As - rhs|| as a measure of solution quality
    double residual_norm = (mat * s - rhs_vector).norm();
    std::cout << "residual norm ||As-rhs||: " << residual_norm << std::endl;

    // Compare: Eta_next from C solver should equal s + Eta for the first row
    std::cout << "Comparing C solver (psi) vs Eigen GMRES solution for first row:" << std::endl;
    for (int i = 0; i < WIDTH; ++i) {
        size_t idx = rowmaj_idx(i, 0, 0);
        double expected = s[i];
        EXPECT_NEAR(psi.p[idx], expected, 1e-10) 
            << "Mismatch at i=" << i << " (grid index " << idx << ")"
            << ": C solver psi = " << psi.p[idx] << ", Eigen = " << expected;
    }

    free_pressure(&rhs_matrix);
    cleanup_test_fields(&U_next, &psi, &phi_lower, &phi_higher, &pressure);
}