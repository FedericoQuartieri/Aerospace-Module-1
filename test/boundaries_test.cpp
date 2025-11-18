#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
// GMRES solver is in the unsupported IterativeSolvers module
#include <unsupported/Eigen/IterativeSolvers>
#include <Eigen/Dense>

extern "C" {
#include "g_field.h"
#include "force_field.h"
#include "pressure.h"
#include "velocity_field.h"
#include "tridiagonal_blocks.h"
#include "constants.h"
#include "utils.h"
}


#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>



// Funzione helper: costruisce sistema tridiagonale Eigen con BC
Eigen::VectorXd build_and_solve_eigen(
    const std::vector<DTYPE>& w,
    const std::vector<DTYPE>& f,
    const std::vector<DTYPE>& uBC_left,
    const std::vector<DTYPE>& uBC_d2,
    const std::vector<DTYPE>& uBC_d3,
    DTYPE delta)
{
    int n = f.size();
    // Utilizziamo un formato sparso per maggiore efficienza con matrici tridiagonali
    // Tuttavia, per la dimostrazione e il piccolo n=50, MatrixXd va bene.
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd rhs(n);

    // Copio l'RHS originale (f)
    for(int i=0;i<n;i++)
        rhs(i) = f[i];

    // Costruisco la matrice A interna (da i=1 a i=n-2)
    // Diagonal: (1 - 2*w[i]), Off-diagonals: w[i]
    for(int i=1;i<n-1;i++) {
        A(i,i-1) = w[i];
        A(i,i+1) = w[i];
        A(i,i) = 1.0 - 2.0*w[i];
    }
    
    // LEFT BC (i=0): Il tuo algoritmo impone u[0] = BC
    A.row(0).setZero();
    A(0,0) = 1.0;
    // Il RHS è calcolato dalla tua funzione Thomas:
    rhs(0) = uBC_left[0] - 0.5 * delta * (uBC_d2[0] + uBC_d3[0]);

    // RIGHT BC (i=n-1): Il tuo algoritmo impone u[n-1] = BC
    A.row(n-1).setZero();
    A(n-1,n-1) = 1.0;
    // Il RHS è calcolato dalla tua funzione Thomas:
    rhs(n-1) = uBC_left[n-1] - 0.5 * delta * (uBC_d2[n-1] + uBC_d3[n-1]);

    // Solve
    return A.lu().solve(rhs);
}

TEST(ThomasTest, CompareWithEigenLargeSystem)
{
    const int n = 50;
    const DTYPE delta = 0.1;

    std::vector<DTYPE> w(n, 0.1);         // coefficiente off-diagonale
    std::vector<DTYPE> f(n);              // RHS interno (poi sovrascritto ai nodi 0 e n-1)
    std::vector<DTYPE> tmp(n, 0.0);       // per Thomas
    std::vector<DTYPE> u(n, 0.0);         // soluzione Thomas
    
    // Tutti questi vettori devono avere dimensione n per l'accesso ai puntatori:
    std::vector<DTYPE> uBC(n, 0.0);       // Valori u per BC (letti solo a 0 e n-1)
    std::vector<DTYPE> d2(n, 0.0);        // Derivata seconda per BC (letta solo a 0 e n-1)
    std::vector<DTYPE> d3(n, 0.0);        // Derivata terza per BC (letta solo a 0 e n-1)

    // RHS arbitrario INTERNO (da i=1 a i=n-2)
    for(int i=0;i<n;i++)
        f[i] = static_cast<DTYPE>(i+1); 

    // Imposto i valori significativi delle BC
    uBC[0] = 1.0;
    uBC[n-1] = 2.0;

    // Derivate tangenziali (arbitrary)
    for(int i=0;i<n;i++){
        d2[i] = 0.5;
        d3[i] = 0.25;
    }

    // Risolvo con Eigen
    Eigen::VectorXd u_eigen = build_and_solve_eigen(w,f,uBC,d2,d3,delta);

    // Risolvo con Thomas
    // f viene modificato in-place durante la forward elimination
    Thomas(w.data(), n, tmp.data(), f.data(), u.data(),
           uBC.data(), d2.data(), d3.data(), delta);

    // Confronto
    for(int i=0;i<n;i++){
        EXPECT_NEAR(u[i], u_eigen(i), 1e-10)
            << "Mismatch at index " << i;
    }
}