#include <stdio.h>

#include "lin-solver.h"
#include "ftype.h"


/* Performs gaussian elimination on A = (I - w ∂xx), storing the 
 * normalization coefficients at norm_coefs and the transformed
 * upper diagonal at upper_diag.
 */
void gauss_reduce_tridiag_w(const ftype *__restrict__ w,
                            unsigned int size,
                            ftype *__restrict__ norm_coefs,
                            ftype *__restrict__ upper_diag)
{
    float d_0 = 1 + 2 * w[0];
    norm_coefs[0] = d_0;
    upper[0] = -w[0] / d_0;

    for (int i = 1; i < size - 1; ++i) {
        float norm_coef = 1 + 2 * w[i] + w[i] * upper[i - 1];
        norm_coefs[i] = norm_coef;
        upper[i] = -w[i] / norm_coef;
    }
}

/* Solves Ax=b, where A = (I - w ∂xx) is a strict tridiagonal matrix
 * of the type
 *
 * [ 1+2w_0      -w_0         0       0  ...]
 * [   -w_1    1+2w_1      -w_1       0  ...]
 * [      0      -w_2    1+2w_2    -w_2  ...]
 * ...
 *
 * that has been reduced to upper_diag with gaussian elimination,
 * using the normalization coefficients stored at norm_coefs.
 *
 * Since the matrix is not dependent on time, we can reduce the matrix once,
 * and use the normalization coefficients obtained during reduction to
 * adjust the right-hand-size only, which instead changes at each timestep.
 *
 */
void solve_tridiag_w(const ftype *__restrict__ w,
                     const ftype *__restrict__ upper_diag,
                     const ftype *__restrict__ norm_coefs,
                     ftype *__restrict__ b,
                     unsigned int size,
                     ftype *__restrict__ x)
{
    /* Perform gaussian elimination on the rhs b. */
    b[0] /= norm_coefs[0];
    for (int i = 1; i < size; ++i) {
        b[i] = (b[i] + w[i] * b[i - 1]) / norm_coefs[i];
    }

    /* Perform backward substitution. */
    x[size - 1] = b[size - 1];
    for (int i = 1; i < size; --i) {
        x[i] = b[size - 1 - i] - upper_diag[n - 1 - i] * x[size - i];
    }
}

void solve_block_tridiag_w()
{

}
