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
    for (int i = 1; i < n; ++i) {
        ftype w_i = w[i];
        ftype norm_coef = 1 / (1 + 2 * w_i + w_i * tmp[i - 1]);
        tmp[i] = -w_i * norm_coef;
        f[i] = (f[i] + w_i * f[i - 1]) * norm_coef;
    }

    /* Perform backward substitution. */
    u[n - 1] = f[n - 1];
    for (int i = 1; i < n; ++i) {
        u[n - i - 1] = f[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
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

/* Solves the block diagonal system (I - ∂yy)u = f. */
void solve_wDyy_tridiag_blocks(const ftype *__restrict__ w,
                               unsigned int depth,
                               unsigned int height,
                               unsigned int width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* We solve for each face of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {

        /* Gauss reduce the first row. */
        size_t face_offset = i * (width * height);
        for (int k = 0; k < width; ++k) {
            size_t idx = face_offset + k;
            ftype w_0 = w[idx];
            ftype d_0 = 1 + 2 * w_0;
            /* Using tmp to store reduced upper diagonal. */
            tmp[idx - face_offset] = -w_0 / d_0;
            f[idx] /= d_0;
        }
        /* Gauss reduce for the whole remaining face, one row at a time. */
        for (int j = 1; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                size_t idx = face_offset + j * width + k;
                ftype w_i = w[idx];
                ftype norm_coef = 1 / (1 + 2 * w_i + w_i * tmp[idx - width]);
                tmp[idx - face_offset] = -w_i * norm_coef;
                f[idx] = (f[idx] + w_i * f[idx - width]) * norm_coef;
            }
        }

        /* Backward substitute the last row. */
        for (int k = 0; k < width; ++k) {
            size_t idx = face_offset + (height - 1) * width + k;
            u[idx] = f[idx];
        }
        /* Backward substitute the remaining face, one row at a time. */
        for (int j = 1; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                size_t idx = face_offset + (height - j - 1) * width + k;
                u[idx] = f[idx] - tmp[idx - face_offset] * u[idx + width];
            }
        }
    }
}

/* Solves the block diagonal system (I - ∂zz)u = f. */
void solve_wDzz_tridiag_blocks(const ftype *__restrict__ w,
                               unsigned int depth,
                               unsigned int height,
                               unsigned int width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* Gauss reduce the first face. */
    for (int j = 0; j < height; ++j) {
        for (int k = 0; k < width; ++k) {
            size_t idx = j * width + k;
            ftype w_0 = w[idx];
            ftype d_0 = 1 + 2 * w_0;
            tmp[idx] = -w_0 / d_0;
            f[idx] /= d_0;
        }
    }
    /* Gauss reduce the whole remaining domain, one face at a time. */
    for (int i = 1; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                size_t idx = i * (width * height) + j * width + k;
                ftype w_i = w[idx];
                ftype norm_coef =
                    1 / (1 + 2 * w_i + w_i * tmp[idx - (height * width)]);
                tmp[idx] = -w_i * norm_coef;
                f[idx] = (f[idx] + w_i * f[idx - height * width]) * norm_coef;
            }
        }
    }

    /* Backward substitute the last face. */
    for (int j = 0; j < height; ++j) {
        for (int k = 0; k < width; ++k) {
            size_t idx = (depth - 1) * (width * height) + j * width + k;
            u[idx] = f[idx];
        }
    }
    /* Backward subsitute the whole remaining domain, one face at a time. */
    for (int i = 1; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                size_t idx = (depth - i - 1) + j * width + k;
                u[idx] = f[idx] - tmp[idx] * u[idx + height * width];
            }
        }
    }
}
