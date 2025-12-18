#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
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
#include <algorithm>


using namespace Eigen;

void solve_Dxx_tridiag_blocks_eigen(DTYPE *Eta_next_component, DTYPE *f_field_component, DTYPE *Gamma, 
                                    DTYPE *__restrict__ u_BC_current_direction,
                                    DTYPE *__restrict__ u_BC_derivative_second_direction,
                                    DTYPE *__restrict__ u_BC_derivative_third_direction) {

    const int N = WIDTH; // Dimensione del sistema 1D
    SparseMatrix<DTYPE> A(N, N);
    
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            size_t off = k * (HEIGHT_GHOST * WIDTH_GHOST) + j * WIDTH_GHOST;
            
            // --- FASE 1: Costruzione della Matrice A e Vettore b per il Blocco ---
            A.setZero();
            VectorXd b(N);
            std::vector<Triplet<DTYPE>> triplets;

            for (int i = 0; i < N; ++i) {
                // Il tuo coefficiente w[i] è -Gamma[i] * DX_INVERSE_SQUARE
                DTYPE w_i = -Gamma[off + i] * DX_INVERSE_SQUARE; 
                
                // Calcola il valore della BC (come nel tuo Thomas)
                DTYPE bc_val = u_BC_current_direction[off + i] - 
                               0.5 * DX * (u_BC_derivative_second_direction[off + i] + u_BC_derivative_third_direction[off + i]);

                if (i == 0 || i == N - 1) {
                    // Nodi di Bordo (A[i,i] = 1.0, b[i] = BC_Value)
                    triplets.push_back(Triplet<DTYPE>(i, i, 1.0));
                    b(i) = bc_val;
                    
                } else {
                    // Nodi Interni (coefficienti dedotti dal tuo Thomas)
                    // Diagonale B_i = (1.0 - 2.0 * w[i])
                    DTYPE Bi = (1.0 - 2.0 * w_i);
                    
                    triplets.push_back(Triplet<DTYPE>(i, i, Bi));
                    // Sottodiagonale A_i = w[i]
                    triplets.push_back(Triplet<DTYPE>(i, i - 1, w_i));
                    // Sopradiagonale C_i = w[i]
                    triplets.push_back(Triplet<DTYPE>(i, i + 1, w_i));

                    b(i) = f_field_component[off + i]; // Lato destro originale
                }
            }
            A.setFromTriplets(triplets.begin(), triplets.end());
            A.makeCompressed();

            // --- FASE 2: Risoluzione del Blocco ---
            SimplicialLDLT<SparseMatrix<DTYPE>> solver;
            solver.compute(A);

            if (solver.info() != Success) {
                std::cerr << "Eigen: Errore nella fattorizzazione nel blocco (" << k << "," << j << ")" << std::endl;
                continue;
            }

            // Mappa per la soluzione (scrive direttamente nell'array C di destinazione)
            Map<VectorXd> x_map(Eta_next_component + off, N);
            
            // Risolvi Ax = b
            x_map = solver.solve(b); 

            if (solver.info() != Success) {
                std::cerr << "Eigen: Errore nella risoluzione nel blocco (" << k << "," << j << ")" << std::endl;
            }
        }
    }
}

TEST(BlockSolverTest, CompareThomasWithEigen)
{
    // --- 1. Allocazione e Inizializzazione dei Dati ---
    
    // I dati saranno allocati come vettori C++ e mappati per le funzioni.
    std::vector<DTYPE> Gamma(GRID_SIZE);
    std::vector<DTYPE> f_field(GRID_SIZE);
    std::vector<DTYPE> u_BC(GRID_SIZE);
    std::vector<DTYPE> d2_BC(GRID_SIZE);
    std::vector<DTYPE> d3_BC(GRID_SIZE);
    
    // Vettori di soluzione: 
    std::vector<DTYPE> Eta_thomas(GRID_SIZE, 0.0);
    std::vector<DTYPE> Eta_eigen(GRID_SIZE, 0.0);
    
    // I vettori f_field sono modificati da Thomas, ne serve una copia.
    std::vector<DTYPE> f_field_thomas_copy(GRID_SIZE);

    // Seed per la generazione di dati casuali
    srand(42); 

    // Popolamento con dati pseudocasuali e BC realistiche
    for(int i = 0; i < GRID_SIZE; i++) {
        // Gamma (coefficiente che influenza w)
        Gamma[i] = 1.0 + (DTYPE)rand() / RAND_MAX * 0.5; 
        
        // Lato Destro (f)
        f_field[i] = (DTYPE)rand() / RAND_MAX * 10.0;
        
        // BCs: fissiamo solo i bordi di ciascuna riga
        // u_BC, d2_BC, d3_BC dovrebbero essere significativi solo ai nodi 0 e N-1 di ogni blocco
        u_BC[i] = 0.0;
        d2_BC[i] = 0.0;
        d3_BC[i] = 0.0;
    }
    
    // Imposta le BCs solo sui bordi X (i=0 e i=N-1) di ciascun blocco (k, j)
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            size_t off = k * (HEIGHT * WIDTH) + j * WIDTH;
            
            // Bordo Sinistro (i=0)
            u_BC[off + 0] = 5.0 + k * 0.1; 
            d2_BC[off + 0] = 1.0;
            d3_BC[off + 0] = -0.5;

            // Bordo Destro (i=N-1)
            u_BC[off + WIDTH - 1] = 10.0 - j * 0.2;
            d2_BC[off + WIDTH - 1] = -1.0;
            d3_BC[off + WIDTH - 1] = 0.5;
        }
    }

    // --- 2. Esecuzione ---

    // Copia il vettore f, dato che solve_Dxx_tridiag_blocks (tramite Thomas) lo modifica
    std::copy(f_field.begin(), f_field.end(), f_field_thomas_copy.begin());
    
    // Esegui la tua funzione originale (Thomas)
    solve_Dxx_tridiag_blocks(Eta_thomas.data(), f_field_thomas_copy.data(), Gamma.data(), 
                            d2_BC.data(), d3_BC.data());

    // Esegui la versione Eigen
    solve_Dxx_tridiag_blocks_eigen(Eta_eigen.data(), f_field.data(), Gamma.data(), 
                                   u_BC.data(), d2_BC.data(), d3_BC.data());

    // --- 3. Confronto ---
    for(int i = 0; i < GRID_SIZE; i++){
        // Confronto di ogni punto della griglia 3D
        EXPECT_NEAR(Eta_thomas[i], Eta_eigen[i], 1e-10)
            << "Mismatch at global index " << i << ". Thomas: " << Eta_thomas[i] << ", Eigen: " << Eta_eigen[i];
    }
}