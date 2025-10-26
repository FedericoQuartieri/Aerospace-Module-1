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
#else

static vftype ONES;
static vftype SIGN_MASK;

#define vinv(vec) vxor(vec, SIGN_MASK)

static inline __attribute__((always_inline))
void transpose_vtile(const ftype *__restrict__ src,
                     uint32_t src_stride,
                     uint32_t dst_stride,
                     ftype *__restrict__ dst)
{
    /* TODO: faster version if you transpose in memory using insert2f128? */
#ifdef FLOAT
    vftype r0 = vload(src + 0 * src_stride);
    vftype r1 = vload(src + 1 * src_stride);
    vftype r2 = vload(src + 2 * src_stride);
    vftype r3 = vload(src + 3 * src_stride);
    vftype r4 = vload(src + 4 * src_stride);
    vftype r5 = vload(src + 5 * src_stride);
    vftype r6 = vload(src + 6 * src_stride);
    vftype r7 = vload(src + 7 * src_stride);
    vtranspose(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    vstore(dst + 0 * dst_stride, r0);
    vstore(dst + 1 * dst_stride, r1);
    vstore(dst + 2 * dst_stride, r2);
    vstore(dst + 3 * dst_stride, r3);
    vstore(dst + 4 * dst_stride, r4);
    vstore(dst + 5 * dst_stride, r5);
    vstore(dst + 6 * dst_stride, r6);
    vstore(dst + 7 * dst_stride, r7);
#else
    vftype r0 = vload(src + 0 * src_stride);
    vftype r1 = vload(src + 1 * src_stride);
    vftype r2 = vload(src + 2 * src_stride);
    vftype r3 = vload(src + 3 * src_stride);
    vtranspose(&r0, &r1, &r2, &r3);
    vstore(dst + 0 * dst_stride, r0);
    vstore(dst + 1 * dst_stride, r1);
    vstore(dst + 2 * dst_stride, r2);
    vstore(dst + 3 * dst_stride, r3);
#endif
}

static inline __attribute__((always_inline))
void gauss_reduce_vstrip_init(const ftype *__restrict__ w,
                              ftype *__restrict__ upper,
                              ftype *__restrict__ f_src,
                              ftype *__restrict__ f_dst)
{
    vftype ws = vload(w);
    vftype fs = vload(f_src);
    vftype ds = vadd(ONES, vadd(ws, ws));
    vftype uppers = vdiv(vinv(ws), ds);
    vstore(upper, uppers);
    vstore(f_dst, vdiv(fs, ds));
}

static inline __attribute__((always_inline))
void gauss_reduce_vstrip(const ftype *__restrict__ w,
                         ftype *__restrict__ upper_prev,
                         const ftype *__restrict__ f_src,
                         ftype *f_prev,
                         ftype *f_dst)
{
    vftype ws = vload(w);
    vftype upper_prevs = vload(upper_prev);
    vftype f_prevs = vload(f_prev);
    vftype fs = vload(f_src);

    vftype norm_coefs = vfmadd(ws, upper_prevs, vadd(ONES, vadd(ws, ws)));

    vstore(upper_prev + VLEN, vdiv(vinv(ws), norm_coefs));
    vstore(f_dst, vdiv(vfmadd(ws, f_prevs, fs), norm_coefs));
}

static inline __attribute__((always_inline))
vftype backward_sub_vstrip(const ftype *__restrict__ f,
                           const ftype *__restrict__ upper,
                           vftype u_prevs,
                           ftype *__restrict__ u)
{
    vftype fs = vload(f);
    vftype uppers = vload(upper);
    vftype u_curr = vfmadd(vinv(uppers), u_prevs, fs);
    vstore(u, u_curr);
    return u_curr;
}

#endif

/* Solves the block diagonal system (I - ∂xx)u = f. */
void solve_wDxx_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               /* tmp buffer of size 2 * (VLEN * width) */
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
    ONES = vbroadcast(1.0);
    SIGN_MASK = vbroadcast(-0.0f);

    ftype *__restrict__ tmp_upp = tmp;
    ftype *__restrict__ tmp_f = tmp + width * VLEN;

    for (int i = 0; i < depth; ++i) {
        /* Solving in groups of VLEN rows. */
        for (int j = 0; j < height; j += VLEN) {
            uint64_t offset = height * width * i + width * j;

            ftype f_t[VLEN * VLEN];
            ftype w_t[VLEN * VLEN];
            /* Load and transpose first tile. */
            transpose_vtile(f + offset, width, VLEN, f_t); 
            transpose_vtile(w + offset, width, VLEN, w_t); 

            /* Reduce first column of the tile. */
            gauss_reduce_vstrip_init(w_t,
                                     tmp_upp,
                                     f_t,
                                     tmp_f);
            /* Reduce remaining columns of the tile. */
            for (int k = 1; k < VLEN; ++k) {
                gauss_reduce_vstrip(w_t + VLEN * k,
                                    tmp_upp + VLEN * (k - 1),
                                    f_t + VLEN * k,
                                    tmp_f + VLEN * (k - 1),
                                    tmp_f + VLEN * k);
            }

            /* Reduce remaining tiles except the last one. */
            for (uint32_t tk = VLEN; tk < width - VLEN; tk += VLEN) {
                /* Load and transpose next tile. */
                transpose_vtile(f + offset + tk, width, VLEN, f_t);
                transpose_vtile(w + offset + tk, width, VLEN, w_t);
                for (uint32_t k = 0; k < VLEN; ++k) {
                    /* TODO: use previous vec f instead of loading again. */
                    gauss_reduce_vstrip(w_t + VLEN * k,
                                        tmp_upp + VLEN * (tk + k - 1),
                                        f_t + VLEN * k,
                                        tmp_f + VLEN * (tk + k - 1),
                                        tmp_f + VLEN * (tk + k));
                }
            }

            transpose_vtile(f + offset + width - VLEN, width, VLEN, f_t);
            transpose_vtile(w + offset + width - VLEN, width, VLEN, w_t);
            /* Reduce last tile. */
            for (int k = 0; k < VLEN; ++k) {
                gauss_reduce_vstrip(w_t + VLEN * k,
                                    tmp_upp + VLEN * (width - VLEN + k - 1),
                                    f_t + VLEN * k,
                                    tmp_f + VLEN * (width - VLEN + k - 1),
                                    tmp_f + VLEN * (width - VLEN + k));
            }

            ftype u_t[VLEN * VLEN] = {0};
            vftype u_last = vbroadcast(0.0f);

            for (uint32_t tk = 0; tk < width; tk += VLEN) {
                for (int k = 0; k < VLEN; ++k) {
                    u_last = backward_sub_vstrip(
                        tmp_f + VLEN * (width - 1 - (tk + k)),
                        tmp_upp + VLEN * (width - 1 - (tk + k)),
                        u_last,
                        u_t + VLEN * (VLEN - 1 - k));
                }
                 /* Transpose and store. */
                transpose_vtile(
                    u_t, VLEN, width, u + offset + width - VLEN - tk);
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
