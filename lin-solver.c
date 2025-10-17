#include "lin-solver.h"

/* Solves Au=f using the Thomas algorithm,
 * where A is a nxn tridiagonal matrix of the type:
 *
 * [ 1+2w_0      -w_0         0       0  ...]
 * [   -w_1    1+2w_1      -w_1       0  ...]
 * [      0      -w_2    1+2w_2    -w_2  ...]
 * ...
 */
static void solve_wDxx_tridiag(const ftype *__restrict__ w,
                               unsigned int n,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* Perform gaussian elimination. */
    ftype d_0 = 1 + 2 * w[0];
    /* Using tmp to store reduced upper diagonal. */
    tmp[0] = -w[0] / d_0;
    f[0] /= d_0;
    for (int i = 1; i < n - 1; ++i) {
        ftype w_i = w[i];
        ftype norm_coef = 1 / (1 + 2 * w_i + w_i * tmp[i - 1]);
        tmp[i] = -w_i * norm_coef;
        f[i] = (f[i] + w_i * f[i - 1]) * norm_coef;
    }

    /* Perform backward substitution. */
    u[n - 1] = f[n - 1];
    for (int i = 1; i < n; ++i) {
        u[i] = f[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}

/* Solves the block diagonal system (I - ∂xx)u = f. */
void solve_wDxx_tridiag_blocks(const ftype *__restrict__ w,
                               unsigned int depth,
                               unsigned int height,
                               unsigned int width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* Solving for each row of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            /* Here we solve for a single block. */
            size_t off = i * (depth * width) + j * width;
            solve_wDxx_tridiag(w + off, width, tmp, f + off, u + off);
        }
    }
}
