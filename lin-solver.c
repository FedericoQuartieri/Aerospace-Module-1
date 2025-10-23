#include <string.h>
#include <stdio.h>

#include "lin-solver.h"

/* Solves Au=f using the Thomas algorithm,
 * where A is a nxn tridiagonal matrix of the type:
 *
 * [ 1+2w_0      -w_0         0       0  ...]
 * [   -w_1    1+2w_1      -w_1       0  ...]
 * [      0      -w_2    1+2w_2    -w_2  ...]
 * ...
 */
#ifdef AUTO_VEC
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
        ftype norm_coef = 1 + 2 * w_i + w_i * tmp[i - 1];
        tmp[i] = -w_i / norm_coef;
        f[i] = (f[i] + w_i * f[i - 1]) / norm_coef;
    }

    /* Perform backward substitution. */
    u[n - 1] = f[n - 1];
    for (int i = 1; i < n; ++i) {
        u[n - i - 1] = f[n - 1 - i] - tmp[n - 1 - i] * u[n - i];
    }
}
#endif

/* Solves the block diagonal system (I - ∂xx)u = f. */
void solve_wDxx_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
#ifdef AUTO_VEC
    /* Solving for each row of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            /* Here we solve for a single block. */
            uint64_t off = height * width * i + width * j;
            solve_wDxx_tridiag(w + off, width, tmp, f + off, u + off);
        }
    }
#else
    vftype ones = vbroadcast(1.0);
    vftype sign_mask = vbroadcast(-0.0f);

    int32_t __attribute__((aligned(32))) offsets[VLEN];
    for (int i = 0; i < VLEN; ++i) {
        offsets[i] = width * i * sizeof(ftype);
    }

    #ifdef FLOAT
    __m256i mask = _mm256_set1_epi32(0xffffffff);
    __m256i gather_off = _mm256_maskload_epi32(offsets, mask);
    #else
    __m128i gather_off = _mm_load_si128((const __m128i *) offsets);
    #endif

    for (int i = 0; i < depth; ++i) {
        /* Solving in groups of VLEN rows. */
        for (int j = 0; j < height; j += VLEN) {
            uint64_t offset = height * width * i + width * j;

            /* Reduce first column of the group. */
            vftype w_0s = vgather(w + offset, gather_off, 1);
            vftype f_0s = vgather(f + offset, gather_off, 1);
            vftype d_0s = vadd(ones, vadd(w_0s, w_0s));
            vftype upper_0s = vdiv(vxor(w_0s, sign_mask), d_0s);
            vscatter(upper_0s, tmp, width);
            vscatter(vdiv(f_0s, d_0s), f + offset, width);

            /* Reduce remaining columns of the group. */
            for (int k = 1; k < width - 1; ++k) {
                /* WARNING: gathers are also unaligned here. */
                vftype ws = vgather(w + offset + k, gather_off, 1);
                vftype upper_prevs = vgather(tmp + k - 1, gather_off, 1);
                vftype f_prevs = vgather(f + offset + k - 1, gather_off, 1);
                vftype fs = vgather(f + offset + k, gather_off, 1);

                vftype norm_coefs = vfmadd(ws, upper_prevs,
                                           vadd(ones, vadd(ws, ws)));

                vscatter(vdiv(vxor(ws, sign_mask),
                              norm_coefs), tmp + k, width);
                vscatter(vdiv(vfmadd(ws, f_prevs, fs),
                              norm_coefs), f + offset + k, width);
            }
            /* Reduce last column. */
            /* TODO: wrap into function. */
            vftype ws = vgather(w + offset + width - 1, gather_off, 1);
            vftype upper_prevs = vgather(tmp + width - 2, gather_off, 1);
            vftype f_prevs = vgather(f + offset + width - 2, gather_off, 1);
            vftype fs = vgather(f + offset + width - 1, gather_off, 1);

            vftype norm_coefs = vfmadd(ws, upper_prevs,
                                       vadd(ones, vadd(ws, ws)));

            vscatter(vdiv(vfmadd(ws, f_prevs, fs),
                          norm_coefs), u + offset + width - 1, width);

            /* Backward substitute, one column of the group at a time. */
            for (int k = 1; k < width; ++k) {
                vftype fs =
                    vgather(f + offset + width - 1 - k, gather_off, 1);
                vftype uppers = vgather(tmp + width - 1 - k, gather_off, 1);
                vftype u_prevs =
                    vgather(u + offset + width - k, gather_off, 1);

                vscatter(vfmadd(vxor(uppers, sign_mask), u_prevs, fs),
                         u + offset + width - 1 - k, width);
            }
        }
    }
#endif
}

static inline __attribute__((always_inline))
void gauss_reduce_row_init(const ftype *__restrict__ w,
                           uint32_t width,
                           ftype *__restrict__ upper,
                           ftype *__restrict__ f)
{
#ifdef AUTO_VEC
    for (uint32_t i = 0; i < width; ++i) {
        ftype w_0 = w[i];
        ftype d_0 = 1 + 2 * w_0;
        upper[i] = -w_0 / d_0;
        f[i] /= d_0;
    }
#else
    vftype ones = vbroadcast(1.0);
    vftype sign_mask = vbroadcast(-0.0f);

    for (uint32_t i = 0; i < width; i += VLEN) {
        vftype ws = vload(w + i);
        vftype fs = vload(f + i);
        vftype ds = vadd(ones, vadd(ws, ws));

        vstore(upper + i, vdiv(vxor(ws, sign_mask), ds));
        vstore(f + i, vdiv(fs, ds));
    }
#endif
}

static inline __attribute__((always_inline))
void gauss_reduce_row(const ftype *__restrict__ w,
                      uint32_t width,
                      uint32_t row_stride,
                      ftype *__restrict__ upper_prev,
                      ftype *f_prev,
                      ftype *f_dst)
{
#ifdef AUTO_VEC
    for (uint32_t i = 0; i < width; ++i) {
        ftype w_i = w[i];
        ftype norm_coef = 1 + 2 * w_i + w_i * upper_prev[i];
        upper_prev[row_stride + i] = -w_i / norm_coef;
        f_dst[i] = (f_prev[row_stride + i] + w_i * f_prev[i]) / norm_coef;
    }
#else
    vftype ones = vbroadcast(1.0);
    /* Used to invert the sign. */
    vftype sign_mask = vbroadcast(-0.0f);

    for (uint32_t i = 0; i < width; i += VLEN) {
        vftype ws = vload(w + i);
        vftype upper_prevs = vload(upper_prev + i);
        vftype f_prevs = vload(f_prev + i);
        vftype fs = vload(f_prev + row_stride + i);

        vftype norm_coefs = vfmadd(ws, upper_prevs,
                                   vadd(ones, vadd(ws, ws)));

        vstore(upper_prev + row_stride + i,
               vdiv(vxor(ws, sign_mask), norm_coefs));
        vstore(f_dst + i,
               vdiv(vfmadd(ws, f_prevs, fs), norm_coefs));
    }
#endif
}

static inline __attribute__((always_inline))
void backward_sub_row(const ftype *__restrict__ f,
                      const ftype *__restrict__ upper,
                      uint32_t width,
                      uint32_t row_stride,
                      ftype *__restrict__ u)
{
#ifdef AUTO_VEC
    for (int k = 0; k < width; ++k) {
        u[k] = f[k] - upper[k] * u[k + row_stride];
    }
#else
    vftype sign_mask = vbroadcast(-0.0f);

    for (int k = 0; k < width; k += VLEN) {
        vftype fs = vload(f + k);
        vftype uppers = vload(upper + k);
        vftype u_prevs = vload(u + row_stride + k);

        vstore(u + k, vfmadd(vxor(uppers, sign_mask), u_prevs, fs));
    }
#endif
}

/* Solves the block diagonal system (I - ∂yy)u = f. */
void solve_wDyy_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* We solve for each face of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {
        /* Gauss reduce the first row. */
        uint64_t face_offset = height * width * i;
        /* Using tmp to store reduced upper diagonal. */
        gauss_reduce_row_init(w + face_offset, width, tmp, f + face_offset);
        /* Gauss reduce the remaining face, one row at a time,
         * except the last one. */
        for (uint32_t j = 1; j < height - 1; ++j) {
            gauss_reduce_row(w + face_offset + j * width,
                             width,
                             width,
                             tmp + (j - 1) * width,
                             f + face_offset + (j - 1) * width,
                             f + face_offset + j * width);
        }
        /* Reduce the last row. */
        gauss_reduce_row(w + face_offset + (height - 1) * width,
                         width,
                         width,
                         tmp + (height - 2) * width,
                         f + face_offset + (height - 2) * width,
                         /* Start backward substitution by writing
                          * directly into u. */
                         u + face_offset + (height - 1) * width);

        /* Backward substitute the remaining face, one row at a time. */
        for (int j = 1; j < height; ++j) {
            uint64_t row_offset = face_offset + (height - j - 1) * width;
            backward_sub_row(f + row_offset,
                             tmp + row_offset - face_offset,
                             width,
                             width,
                             u + row_offset);
        }
    }
}

/* Solves the block diagonal system (I - ∂zz)u = f. */
void solve_wDzz_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u)
{
    /* Gauss reduce the first face. */
    for (uint32_t j = 0; j < height; ++j) {
        uint64_t row_offset = j * width;
        gauss_reduce_row_init(w + row_offset,
                              width,
                              tmp + row_offset,
                              f + row_offset);
    }

    /* Gauss reduce the remaining domain, one face at a time,
     * except the last one. */
    for (uint32_t i = 1; i < depth - 1; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            uint64_t row_offset = (height * width) * i + width * j;
            gauss_reduce_row(w + row_offset,
                             width,
                             height * width,
                             tmp + row_offset - (height * width),
                             f + row_offset - (height * width),
                             f + row_offset);
        }
    }
    /* Reduce the last face, indirectly performing
     * backward substitution on it. */
    uint64_t face_offset = (depth - 1) * (height * width);
    for (uint32_t j = 0; j < height; ++j) {
        gauss_reduce_row(w + face_offset + j * width,
                         width,
                         height * width,
                         tmp + face_offset + j * width - (height * width),
                         f + face_offset + j * width - (height * width),
                         u + face_offset + j * width);
    }

    /* Backward subsitute the remaining domain, one face at a time. */
    for (uint32_t i = 1; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            uint64_t row_offset =
                (height * width) * (depth - i - 1) + width * j;
            backward_sub_row(f + row_offset,
                             tmp + row_offset,
                             width,
                             height * width,
                             u + row_offset);
        }
    }
}
