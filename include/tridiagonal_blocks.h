#ifndef TRIDIAGONAL_BLOCKS_H
#define TRIDIAGONAL_BLOCKS_H

#include <stddef.h>
#include "utils.h"
#include "velocity_field.h"
#include "force_field.h"
#include "g_field.h"
#include "neon_instructions.h"

#include <stdbool.h>

/**
 * @brief Solves the tridiagonal system Au = f, using the Thomas algorithm. 
 *
 * The matrix A is a tridiagonal matrix of size n x n, with the following structure:
 *
 * [ 1+2w_0      -w_0         0       0  ...] [u_0]   [f_0]
 * [   -w_1    1+2w_1      -w_1       0  ...] [u_1] = [f_1]
 * [    ...    ...       ...      ....      ] [...]   [...]
 * [     0     0      -w_n    1+2w_n    -w_n] [u_n]   [f_n]
 *
 * The algorithm reduces the matrix to an upper triangular form and then performs
 * backward substitution to find the solution vector u.
 *
 * @param[in] w    Array of size n containing the gamma_coefficients.
 * @param[in] n    Size of the tridiagonal matrix.
 * @param[in,out] tmp Temporary array of size n used to store intermediate values
 *                    during the forward elimination step. The array `tmp` represents
 *                    the reduced matrix in the form:
 *                    [ 1   tmp[0]   0   ]
 *                    [  0    1   tmp[1] ]
 *                    [      0      1    ]
 * @param[in,out] f   Right-hand side vector of size n. This array is modified during
 *                    the computation and will not retain its original values.
 * @param[out] u   Solution vector of size n.
 */

void Thomas_Algorithm(const DTYPE *__restrict__ w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ rhs,
                               DTYPE *__restrict__ u,
                               bool same_direction
                           );
                               
void Thomas_Pressure(const DTYPE w, 
                               unsigned int n,
                               DTYPE *__restrict__ tmp,
                               DTYPE *__restrict__ f,
                               DTYPE *__restrict__ u
                            );

/* Solves the block diagonal system (I - ∂xx)u = f. */
void solve_Dxx_tridiag_blocks(DTYPE *__restrict__ Eta_next_component, DTYPE *__restrict__ f_field_component, DTYPE *__restrict__ Gamma, const Data *data, bool same_direction, int v_component, int time_step);

/* Solves the block diagonal system (I - ∂yy)u = f. */
void solve_Dyy_tridiag_blocks(DTYPE *Zeta_next_component, DTYPE *f_field_component, DTYPE *Gamma, const Data *data, bool same_direction, int v_component, int time_step);

/* Solves the block diagonal system (I - ∂zz)u = f. */
void solve_Dzz_tridiag_blocks(DTYPE *U_next_component, DTYPE *f_field_component, DTYPE *Gamma, const Data *data, bool same_direction, int v_component, int time_step);

void simd_Thomas_Algorithm(DTYPE *__restrict__ simd_w, unsigned int n, DTYPE *__restrict__ simd_tmp, DTYPE *__restrict__ simd_rhs, DTYPE *__restrict__ simd_u, bool same_direction);
void vectorized_solve_Dyy_tridiag_blocks(DTYPE *__restrict__ Zeta_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step);
void vectorized_solve_Dzz_tridiag_blocks(DTYPE *__restrict__ U_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step);

void optimize_solve_Dxx_tridiag_blocks(DTYPE *__restrict__ Eta_prev, DTYPE *__restrict__ Zeta_prev, DTYPE *__restrict__ U_prev, DTYPE *__restrict__ pressure_star, DTYPE *__restrict__ Gamma, const Data *data, bool same_direction, int v_component, int time_step);
void optimized_simd_solve_Dyy_tridiag_blocks(DTYPE *__restrict__ Zeta_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step);
void optimized_simd_solve_Dzz_tridiag_blocks(DTYPE *__restrict__ U_next, DTYPE *__restrict__ rhs, DTYPE *__restrict__ Gamma, const Data *__restrict__ data, bool same_direction, int v_component, int time_step);

#endif
