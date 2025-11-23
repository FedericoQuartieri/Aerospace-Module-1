#include <string.h>
#include <stdio.h>

#include "lin-solver.h"
#include "boundary.h"

#ifdef AUTO_VEC
    #define vneg(v) (-(v))
#else
    #define vneg(vec) vxor(vec, SIGN_MASK)
#endif

DECLARE_BC_U(BC_LEFT)
DECLARE_BC_U(BC_RIGHT)
DECLARE_BC_U(BC_TOP)
DECLARE_BC_U(BC_BOTTOM)
DECLARE_BC_U(BC_FRONT)
DECLARE_BC_U(BC_BACK)

/* TODO: Remove these. */
static vftype ZEROS;
static vftype ONES;
static vftype SIGN_MASK;


/* WARNING: You must scale w by 1/dx^2!! */

static inline __attribute__((always_inline))
void apply_start_bc(vftype u0_x,
                    vftype u0_y,
                    vftype u0_z,
                    ftype *__restrict__ upper,
                    ftype *__restrict__ f_x,
                    ftype *__restrict__ f_y,
                    ftype *__restrict__ f_z)
{
    /* u_x = u_0 + (-du_y/dy -du_z/dz) * dx/2
     * u_y = u0_y
     * u_z = u0_z */

    /* Set upper coefficient to 0 and enforce solution in rhs. */
    vstore(upper, ZEROS);
    vstore(f_x, u0_x); /* du_y/dy = du_z/dz = 0 */
    vstore(f_y, u0_y);
    vstore(f_z, u0_z);
}

static inline __attribute__((always_inline))
vftype compute_end_bc_tang_u(vftype ws,
                             vftype ws2,
                             vftype fs_prev,
                             vftype fs,
                             vftype uns,
                             vftype norm_coeffs)
{
    return vdiv(vsub(vfmadd(fs_prev, ws, vneg(fs)),
                     vmul(ws2, uns)),
                norm_coeffs);
}

static inline __attribute__((always_inline))
void apply_end_bc(const ftype *__restrict__ w,
                  const ftype *__restrict__ upper_prev,
                  const ftype *__restrict__ f_y_prev,
                  const ftype *__restrict__ f_z_prev,
                  const ftype *__restrict__ f_y,
                  const ftype *__restrict__ f_z,
                  vftype un_x,
                  vftype un_y,
                  vftype un_z,
                  vftype *__restrict__ u_x,
                  vftype *__restrict__ u_y,
                  vftype *__restrict__ u_z)
{
    /* u_x = un_x
     * u_y =  (-2w_i un_y - f_y_i + w_i f_y_i_prev) /
     *        (1 + 3w_i + w_i upper_prev_i)
     * u_z =  (-2w_i un_z - f_z_i + w_i f_z_i_prev) /
     *        (1 + 3w_i + w_i upper_prev_i) */

    vftype ws = vload(w);
    vftype upper_prevs = vload(upper_prev);
    vftype fs_y_prevs = vload(f_y_prev);
    vftype fs_z_prevs = vload(f_z_prev);
    vftype fs_y = vload(f_y);
    vftype fs_z = vload(f_z);
    vftype ws2 = vadd(ws, ws);
    vftype norm_coeffs = vfmadd(upper_prevs, ws,
                                vadd(ONES, vadd(ws2, ws)));

    *u_x = un_x;
    *u_y = compute_end_bc_tang_u(ws, ws2, fs_y_prevs,
                                 fs_y, un_y, norm_coeffs);
    *u_z = compute_end_bc_tang_u(ws, ws2, fs_z_prevs,
                                 fs_z, un_z, norm_coeffs);
}


static inline __attribute__((always_inline))
void apply_left_bc(uint32_t x,
                   uint32_t y,
                   uint32_t z,
                   ftype *__restrict__ upper,
                   ftype *__restrict__ f_x,
                   ftype *__restrict__ f_y,
                   ftype *__restrict__ f_z)
{
    vftype u0_x, u0_y, u0_z;
    _get_left_bc_u(x, y, z, &u0_x, &u0_y, &u0_z);

    apply_start_bc(u0_x, u0_y, u0_z, upper, f_x, f_y, f_z);
}

static inline __attribute__((always_inline))
void apply_right_bc(const ftype *__restrict__ w,
                    const ftype *__restrict__ upper_prev,
                    const ftype *__restrict__ f_y_prev,
                    const ftype *__restrict__ f_z_prev,
                    uint32_t x,
                    uint32_t y,
                    uint32_t z,
                    ftype *__restrict__ f_x,
                    ftype *__restrict__ f_y,
                    ftype *__restrict__ f_z,
                    vftype *__restrict__ u_x,
                    vftype *__restrict__ u_y,
                    vftype *__restrict__ u_z)
{
    vftype un_x, un_y, un_z;
    _get_right_bc_u(x, y, z, &un_x, &un_y, &un_z);

    apply_end_bc(w, upper_prev, f_y_prev, f_z_prev,
                 f_y, f_z, un_x, un_y, un_z, u_x, u_y, u_z);

    vstore(f_x, *u_x);
    vstore(f_y, *u_y);
    vstore(f_z, *u_z);
}

#ifdef AUTO_VEC

/* Solves Au=f using the Thomas algorithm,
 * where A is a nxn tridiagonal matrix of the type:
 *
 * [ 1+2w_0      -w_0         0       0  ...]
 * [   -w_1    1+2w_1      -w_1       0  ...]
 * [      0      -w_2    1+2w_2    -w_2  ...]
 * ...
 */
static void solve_wDxx_tridiag(const ftype *__restrict__ w,
                               uint32_t y,
                               uint32_t z,
                               uint32_t n,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f_x,
                               ftype *__restrict__ f_y,
                               ftype *__restrict__ f_z,
                               ftype *__restrict__ u_x,
                               ftype *__restrict__ u_y,
                               ftype *__restrict__ u_z)
{
    /* Perform gaussian elimination. */
    /* Using tmp to store reduced upper diagonal. */

    /* Left boundary conditions! */
    tmp[0] = 0;
    _get_left_bc_u(0, y, z, &f_x[0], &f_y[0], &f_z[0]);

    for (int i = 1; i < n - 1; ++i) {
        ftype w_i = w[i];
        ftype norm_coef = 1 + 2 * w_i + w_i * tmp[i - 1];
        tmp[i] = -w_i / norm_coef;
        f_x[i] = (f_x[i] + w_i * f_x[i - 1]) / norm_coef;
        f_y[i] = (f_y[i] + w_i * f_y[i - 1]) / norm_coef;
        f_z[i] = (f_z[i] + w_i * f_z[i - 1]) / norm_coef;
    }

    /* Right boundary conditions! */
    ftype un_y, un_z;
    _get_right_bc_u(0, y, z, &u_x[n - 1], &un_y, &un_z);
    ftype w_n = w[n - 1];
    ftype norm_coeff = 1 + 3 * w_n + w_n * tmp[n - 2];
    u_y[n - 1] = (-2 * w_n * un_y - f_y[n - 1] + w_n * f_y[n - 2]) /
                 norm_coeff;
    u_z[n - 1] = (-2 * w_n * un_z - f_z[n - 1] + w_n * f_z[n - 2]) /
                 norm_coeff;

    /* Perform backward substitution. */
    for (int i = 1; i < n; ++i) {
        ftype tmp_i = tmp[n - 1 - i];
        u_x[n - i - 1] = f_x[n - 1 - i] - tmp_i * u_x[n - i];
        u_y[n - i - 1] = f_y[n - 1 - i] - tmp_i * u_y[n - i];
        u_z[n - i - 1] = f_z[n - 1 - i] - tmp_i * u_z[n - i];
    }
}

#else

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
void gauss_reduce_vstrip(const ftype *__restrict__ w,
                         ftype *__restrict__ upper_prev,
                         const ftype *__restrict__ f_x_src,
                         const ftype *__restrict__ f_y_src,
                         const ftype *__restrict__ f_z_src,
                         ftype *__restrict__ f_x_dst,
                         ftype *__restrict__ f_y_dst,
                         ftype *__restrict__ f_z_dst)
{
    vftype ws = vload(w);
    vftype upper_prevs = vload(upper_prev);
    vftype f_x_prevs = vload(f_x_dst - VLEN);
    vftype f_y_prevs = vload(f_y_dst - VLEN);
    vftype f_z_prevs = vload(f_z_dst - VLEN);
    vftype fs_x = vload(f_x_src);
    vftype fs_y = vload(f_y_src);
    vftype fs_z = vload(f_z_src);
    vftype norm_coefs = vfmadd(ws, upper_prevs, vadd(ONES, vadd(ws, ws)));
    vstore(upper_prev + VLEN, vdiv(vneg(ws), norm_coefs));
    vstore(f_x_dst, vdiv(vfmadd(ws, f_x_prevs, fs_x), norm_coefs));
    vstore(f_y_dst, vdiv(vfmadd(ws, f_y_prevs, fs_y), norm_coefs));
    vstore(f_z_dst, vdiv(vfmadd(ws, f_z_prevs, fs_z), norm_coefs));
}

static inline __attribute__((always_inline))
void backward_sub_vstrip(const ftype *__restrict__ f_x,
                         const ftype *__restrict__ f_y,
                         const ftype *__restrict__ f_z,
                         const ftype *__restrict__ upper,
                         vftype *__restrict__ u_x_prevs,
                         vftype *__restrict__ u_y_prevs,
                         vftype *__restrict__ u_z_prevs,
                         ftype *__restrict__ u_x,
                         ftype *__restrict__ u_y,
                         ftype *__restrict__ u_z)
{
    vftype fs_x = vload(f_x);
    vftype fs_y = vload(f_y);
    vftype fs_z = vload(f_z);
    vftype uppers = vload(upper);
    *u_x_prevs = vfmadd(vneg(uppers), *u_x_prevs, fs_x);
    vstore(u_x, *u_x_prevs);
    *u_y_prevs = vfmadd(vneg(uppers), *u_y_prevs, fs_y);
    vstore(u_y, *u_y_prevs);
    *u_z_prevs = vfmadd(vneg(uppers), *u_z_prevs, fs_z);
    vstore(u_z, *u_z_prevs);
}

#endif

/* Solves the block diagonal system (I - w∂xx)u = f. */
void solve_wDxx_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               /* tmp buffer of size 4 * (VLEN * width) */
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f_x,
                               ftype *__restrict__ f_y,
                               ftype *__restrict__ f_z,
                               ftype *__restrict__ u_x,
                               ftype *__restrict__ u_y,
                               ftype *__restrict__ u_z)
{
#ifdef AUTO_VEC
    /* Solving for each row of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            /* Here we solve for a single block. */
            uint64_t off = height * width * i + width * j;
            solve_wDxx_tridiag(w + off, j, i, width, tmp,
                               f_x + off, f_y + off, f_z + off,
                               u_x + off, u_y + off, u_z + off);
        }
    }
#else
    ZEROS = vbroadcast(0.0);
    ONES = vbroadcast(1.0);
    SIGN_MASK = vbroadcast(-0.0f);

    ftype *__restrict__ tmp_upp = tmp;
    /* WARNING: Cache aliasing? */
    ftype *__restrict__ tmp_f_x = tmp + 1 * width * VLEN;
    ftype *__restrict__ tmp_f_y = tmp + 2 * width * VLEN;
    ftype *__restrict__ tmp_f_z = tmp + 3 * width * VLEN;

    for (uint32_t i = 0; i < depth; ++i) {
        /* Solving in groups of VLEN rows. */
        for (uint32_t j = 0; j < height; j += VLEN) {
            uint64_t offset = height * width * i + width * j;

            ftype __attribute__((aligned(32))) f_x_t[VLEN * VLEN];
            ftype __attribute__((aligned(32))) f_y_t[VLEN * VLEN];
            ftype __attribute__((aligned(32))) f_z_t[VLEN * VLEN];
            ftype __attribute__((aligned(32))) w_t[VLEN * VLEN];
            /* Load and transpose first tile. */
            transpose_vtile(f_x + offset, width, VLEN, f_x_t); 
            transpose_vtile(f_y + offset, width, VLEN, f_y_t); 
            transpose_vtile(f_z + offset, width, VLEN, f_z_t); 
            transpose_vtile(w + offset, width, VLEN, w_t); 

            /* Apply BCs on the first column of the tile. */
            apply_left_bc(0, j, i, tmp_upp, tmp_f_x, tmp_f_y, tmp_f_z);
            /* Reduce remaining columns of the tile. */
            for (int k = 1; k < VLEN; ++k) {
                gauss_reduce_vstrip(w_t + VLEN * k,
                                    tmp_upp + VLEN * (k - 1),
                                    f_x_t + VLEN * k,
                                    f_y_t + VLEN * k,
                                    f_z_t + VLEN * k,
                                    tmp_f_x + VLEN * k,
                                    tmp_f_y + VLEN * k,
                                    tmp_f_z + VLEN * k);
            }

            /* Reduce remaining tiles except the last one. */
            for (uint32_t tk = VLEN; tk < width - VLEN; tk += VLEN) {
                /* Load and transpose next tile. */
                transpose_vtile(f_x + offset + tk, width, VLEN, f_x_t);
                transpose_vtile(f_y + offset + tk, width, VLEN, f_y_t);
                transpose_vtile(f_z + offset + tk, width, VLEN, f_z_t);
                transpose_vtile(w + offset + tk, width, VLEN, w_t);
                for (uint32_t k = 0; k < VLEN; ++k) {
                    /* TODO: use previous vec f instead of loading again. */
                    gauss_reduce_vstrip(w_t + VLEN * k,
                                        tmp_upp + VLEN * (tk + k - 1),
                                        f_x_t + VLEN * k,
                                        f_y_t + VLEN * k,
                                        f_z_t + VLEN * k,
                                        tmp_f_x + VLEN * (tk + k),
                                        tmp_f_y + VLEN * (tk + k),
                                        tmp_f_z + VLEN * (tk + k));
                }
            }

            transpose_vtile(f_x + offset + width - VLEN, width, VLEN, f_x_t);
            transpose_vtile(f_y + offset + width - VLEN, width, VLEN, f_y_t);
            transpose_vtile(f_z + offset + width - VLEN, width, VLEN, f_z_t);
            transpose_vtile(w + offset + width - VLEN, width, VLEN, w_t);
            /* Reduce last tile except last column. */
            for (int k = 0; k < VLEN - 1; ++k) {
                gauss_reduce_vstrip(w_t + VLEN * k,
                                    tmp_upp + VLEN * (width - VLEN + k - 1),
                                    f_x_t + VLEN * k,
                                    f_y_t + VLEN * k,
                                    f_z_t + VLEN * k,
                                    tmp_f_x + VLEN * (width - VLEN + k),
                                    tmp_f_y + VLEN * (width - VLEN + k),
                                    tmp_f_z + VLEN * (width - VLEN + k));
            }

            vftype u_x_prev, u_y_prev, u_z_prev;
            /* Apply BCs on the right column. */
            apply_right_bc(w_t + VLEN * (VLEN - 1),
                           tmp_upp + VLEN * (width - 2),
                           tmp_f_y + VLEN * (width - 2),
                           tmp_f_z + VLEN * (width - 2),
                           width - 1, j, i,
                           /* Write solutions into f_t buffers,
                            * we will reuse them for u_t buffers */
                           f_x_t + VLEN * (VLEN - 1),
                           f_y_t + VLEN * (VLEN - 1),
                           f_z_t + VLEN * (VLEN - 1),
                           &u_x_prev,
                           &u_y_prev,
                           &u_z_prev);

            /* Reuse local buffers. */
            ftype __attribute__((aligned(32))) *u_x_t = f_x_t;
            ftype __attribute__((aligned(32))) *u_y_t = f_y_t;
            ftype __attribute__((aligned(32))) *u_z_t = f_z_t;

            /* Backward substitute last tile (last col already solved). */
            for (int k = 1; k < VLEN; ++k) {
                backward_sub_vstrip(
                    tmp_f_x + VLEN * (width - 1 - k),
                    tmp_f_y + VLEN * (width - 1 - k),
                    tmp_f_z + VLEN * (width - 1 - k),
                    tmp_upp + VLEN * (width - 1 - k),
                    &u_x_prev,
                    &u_y_prev,
                    &u_z_prev,
                    u_x_t + VLEN * (VLEN - 1 - k),
                    u_y_t + VLEN * (VLEN - 1 - k),
                    u_z_t + VLEN * (VLEN - 1 - k));
            }
            transpose_vtile(u_x_t, VLEN, width, u_x + offset + width - VLEN);
            transpose_vtile(u_y_t, VLEN, width, u_y + offset + width - VLEN);
            transpose_vtile(u_z_t, VLEN, width, u_z + offset + width - VLEN);

            /* Backward substitute one tile at a time. */
            for (uint32_t tk = VLEN; tk < width; tk += VLEN) {
                for (int k = 0; k < VLEN; ++k) {
                    backward_sub_vstrip(
                        tmp_f_x + VLEN * (width - 1 - (tk + k)),
                        tmp_f_y + VLEN * (width - 1 - (tk + k)),
                        tmp_f_z + VLEN * (width - 1 - (tk + k)),
                        tmp_upp + VLEN * (width - 1 - (tk + k)),
                        &u_x_prev,
                        &u_y_prev,
                        &u_z_prev,
                        u_x_t + VLEN * (VLEN - 1 - k),
                        u_y_t + VLEN * (VLEN - 1 - k),
                        u_z_t + VLEN * (VLEN - 1 - k));
                }
                 /* Transpose and store. */
                transpose_vtile(u_x_t, VLEN, width,
                                u_x + offset + width - VLEN - tk);
                transpose_vtile(u_y_t, VLEN, width,
                                u_y + offset + width - VLEN - tk);
                transpose_vtile(u_z_t, VLEN, width,
                                u_z + offset + width - VLEN - tk);
            }
        }
    }
#endif
}

static inline __attribute__((always_inline))
void gauss_reduce_row(const ftype *__restrict__ w,
                      uint32_t width,
                      uint32_t row_stride,
                      ftype *__restrict__ upper,
                      ftype *__restrict__ f_x,
                      ftype *__restrict__ f_y,
                      ftype *__restrict__ f_z)
{
    for (uint32_t i = 0; i < width; i += VLEN) {
        vftype ws = vload(w + i);
        vftype upper_prevs = vload(upper - row_stride + i);
        vftype f_x_prevs = vload(f_x - row_stride + i);
        vftype f_y_prevs = vload(f_y - row_stride + i);
        vftype f_z_prevs = vload(f_z - row_stride + i);
        vftype fs_x = vload(f_x + i);
        vftype fs_y = vload(f_y + i);
        vftype fs_z = vload(f_z + i);
        vftype norm_coefs = vfmadd(ws, upper_prevs, vadd(ONES, vadd(ws, ws)));
        vstore(upper + i, vdiv(vneg(ws), norm_coefs));
        vstore(f_x + i, vdiv(vfmadd(ws, f_x_prevs, fs_x), norm_coefs));
        vstore(f_y + i, vdiv(vfmadd(ws, f_y_prevs, fs_y), norm_coefs));
        vstore(f_z + i, vdiv(vfmadd(ws, f_z_prevs, fs_z), norm_coefs));
    }
}

static inline __attribute__((always_inline))
void backward_sub_row(const ftype *__restrict__ f_x,
                      const ftype *__restrict__ f_y,
                      const ftype *__restrict__ f_z,
                      const ftype *__restrict__ upper,
                      uint32_t width,
                      uint32_t row_stride,
                      ftype *__restrict__ u_x,
                      ftype *__restrict__ u_y,
                      ftype *__restrict__ u_z)
{
    for (int k = 0; k < width; k += VLEN) {
        vftype fs_x = vload(f_x + k);
        vftype fs_y = vload(f_y + k);
        vftype fs_z = vload(f_z + k);
        vftype uppers = vload(upper + k);
        vftype u_x_prevs = vload(u_x + row_stride + k);
        vftype u_y_prevs = vload(u_y + row_stride + k);
        vftype u_z_prevs = vload(u_z + row_stride + k);
        vstore(u_x + k, vfmadd(vneg(uppers), u_x_prevs, fs_x));
        vstore(u_y + k, vfmadd(vneg(uppers), u_y_prevs, fs_y));
        vstore(u_z + k, vfmadd(vneg(uppers), u_z_prevs, fs_z));
    }
}

static inline __attribute__((always_inline))
void apply_top_bc(uint32_t x,
                  uint32_t y,
                  uint32_t z,
                  ftype *__restrict__ upper,
                  ftype *__restrict__ f_x,
                  ftype *__restrict__ f_y,
                  ftype *__restrict__ f_z)
{
    vftype u0_x, u0_y, u0_z;
    _get_top_bc_u(x, y, z, &u0_x, &u0_y, &u0_z);

    apply_start_bc(u0_y, u0_x, u0_z, upper, f_y, f_x, f_z);
}

static inline __attribute__((always_inline))
void apply_bottom_bc(const ftype *__restrict__ w,
                     const ftype *__restrict__ upper_prev,
                     const ftype *__restrict__ f_x_prev,
                     const ftype *__restrict__ f_z_prev,
                     const ftype *__restrict__ f_x,
                     const ftype *__restrict__ f_z,
                     uint32_t x,
                     uint32_t y,
                     uint32_t z,
                     ftype *__restrict__ u_x,
                     ftype *__restrict__ u_y,
                     ftype *__restrict__ u_z)
{
    vftype un_x, un_y, un_z;
    _get_bottom_bc_u(x, y, z, &un_x, &un_y, &un_z);

    vftype _un_x, _un_y, _un_z;
    apply_end_bc(w, upper_prev, f_x_prev, f_z_prev, f_x, f_z,
                 un_y, un_x, un_z, &_un_y, &_un_x, &_un_z);

    vstore(u_x, _un_x);
    vstore(u_y, _un_y);
    vstore(u_z, _un_z);
}

/* Solves the block diagonal system (I - w∂yy)u = f. */
void solve_wDyy_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f_x,
                               ftype *__restrict__ f_y,
                               ftype *__restrict__ f_z,
                               ftype *__restrict__ u_x,
                               ftype *__restrict__ u_y,
                               ftype *__restrict__ u_z)
{
    ZEROS = vbroadcast(0.0);
    ONES = vbroadcast(1.0);
    SIGN_MASK = vbroadcast(-0.0f);

    /* We solve for each face of the domain, one at a time. */
    for (int i = 0; i < depth; ++i) {
        /* Gauss reduce the first row. */
        uint64_t face_offset = height * width * i;

        /* Apply BCs on the first row of the domain. */
        for (uint32_t k = 0; k < width; k += VLEN) {
            apply_top_bc(k, 0, i, tmp + k,
                         f_x + face_offset + k,
                         f_y + face_offset + k,
                         f_z + face_offset + k);
        }
        /* Gauss reduce the remaining face, one row at a time,
         * except the last one. */
        for (uint32_t j = 1; j < height - 1; ++j) {
            gauss_reduce_row(w + face_offset + width * j,
                             width,
                             width,
                             tmp + width * j,
                             f_x + face_offset + width * j,
                             f_y + face_offset + width * j,
                             f_z + face_offset + width * j);
        }
        /* Apply BCs on the last row. */
        for (uint32_t k = 0; k < width; k += VLEN) {
            apply_bottom_bc(w + face_offset + width * (height - 1) + k,
                            tmp + width * (height - 2) + k,
                            f_x + face_offset + width * (height - 2) + k,
                            f_z + face_offset + width * (height - 2) + k,
                            f_x + face_offset + width * (height - 1) + k,
                            f_z + face_offset + width * (height - 1) + k,
                            k, height - 1, i,
                            u_x + face_offset + width * (height - 1) + k,
                            u_y + face_offset + width * (height - 1) + k,
                            u_z + face_offset + width * (height - 1) + k);
        }

        /* Backward substitute the remaining face, one row at a time. */
        for (int j = 1; j < height; ++j) {
            uint64_t row_offset = face_offset + width * (height - j - 1);
            backward_sub_row(f_x + row_offset,
                             f_y + row_offset,
                             f_z + row_offset,
                             tmp + row_offset - face_offset,
                             width,
                             width,
                             u_x + row_offset,
                             u_y + row_offset,
                             u_z + row_offset);
        }
    }
}

static inline __attribute__((always_inline))
void apply_front_bc(uint32_t x,
                    uint32_t y,
                    uint32_t z,
                    ftype *__restrict__ upper,
                    ftype *__restrict__ f_x,
                    ftype *__restrict__ f_y,
                    ftype *__restrict__ f_z)
{
    vftype u0_x, u0_y, u0_z;
    _get_front_bc_u(x, y, z, &u0_x, &u0_y, &u0_z);

    apply_start_bc(u0_z, u0_x, u0_y, upper, f_z, f_x, f_y);
}

static inline __attribute__((always_inline))
void apply_back_bc(const ftype *__restrict__ w,
                   const ftype *__restrict__ upper_prev,
                   const ftype *__restrict__ f_x_prev,
                   const ftype *__restrict__ f_y_prev,
                   const ftype *__restrict__ f_x,
                   const ftype *__restrict__ f_y,
                   uint32_t x,
                   uint32_t y,
                   uint32_t z,
                   ftype *__restrict__ u_x,
                   ftype *__restrict__ u_y,
                   ftype *__restrict__ u_z)
{
    vftype un_x, un_y, un_z;
    _get_back_bc_u(x, y, z, &un_x, &un_y, &un_z);

    vftype _un_x, _un_y, _un_z;
    apply_end_bc(w, upper_prev, f_x_prev, f_y_prev, f_x, f_y,
                 un_z, un_x, un_y, &_un_z, &_un_x, &_un_y);

    vstore(u_x, _un_x);
    vstore(u_y, _un_y);
    vstore(u_z, _un_z);
}

/* Solves the block diagonal system (I - w∂zz)u = f. */
void solve_wDzz_tridiag_blocks(const ftype *__restrict__ w,
                               uint32_t depth,
                               uint32_t height,
                               uint32_t width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f_x,
                               ftype *__restrict__ f_y,
                               ftype *__restrict__ f_z,
                               ftype *__restrict__ u_x,
                               ftype *__restrict__ u_y,
                               ftype *__restrict__ u_z)
{
    ZEROS = vbroadcast(0.0);
    ONES = vbroadcast(1.0);
    SIGN_MASK = vbroadcast(-0.0f);

    /* Apply BCs to the first face. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; k += VLEN) {
            apply_front_bc(k, j, 0,
                           tmp + width * j + k,
                           f_x + width * j + k,
                           f_y + width * j + k,
                           f_z + width * j + k);
        }
    }

    /* Gauss reduce the remaining domain, one face at a time,
     * except the last one. */
    for (uint32_t i = 1; i < depth - 1; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            uint64_t row_offset = height * width * i + width * j;
            gauss_reduce_row(w + row_offset,
                             width,
                             height * width,
                             tmp + row_offset,
                             f_x + row_offset,
                             f_y + row_offset,
                             f_z + row_offset);
        }
    }
    /* Apply BCs the last face, solving directly. */
    uint64_t face_offset = height * width * (depth - 1);
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; k += VLEN) {
            apply_back_bc(w + face_offset + width * j + k,
                          tmp + height * width * (depth - 2) + width * j + k,
                          f_x + height * width * (depth - 2) + width * j + k,
                          f_y + height * width * (depth - 2) + width * j + k,
                          f_x + face_offset + width * j + k,
                          f_y + face_offset + width * j + k,
                          k, j, depth - 1,
                          u_x + face_offset + width * j + k,
                          u_y + face_offset + width * j + k,
                          u_z + face_offset + width * j + k);
        }
    }

    /* Backward subsitute the remaining domain, one face at a time. */
    for (uint32_t i = 1; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            uint64_t row_offset =
                height * width * (depth - i - 1) + width * j;
            backward_sub_row(f_x + row_offset,
                             f_y + row_offset,
                             f_z + row_offset,
                             tmp + row_offset,
                             width,
                             height * width,
                             u_x + row_offset,
                             u_y + row_offset,
                             u_z + row_offset);
        }
    }
}


#define load_vtile(src, stride,      \
                   r1, r2, r3, r4,   \
                   r5, r6, r7, r8)   \
do {                                 \
    r1 = vload(src + 0 * stride);    \
    r2 = vload(src + 1 * stride);    \
    r3 = vload(src + 2 * stride);    \
    r4 = vload(src + 3 * stride);    \
    r5 = vload(src + 4 * stride);    \
    r6 = vload(src + 5 * stride);    \
    r7 = vload(src + 6 * stride);    \
    r8 = vload(src + 7 * stride);    \
} while (0)

#define fin_diff(r1, r2, r3, r4, \
                 r5, r6, r7, r8) \
do {                             \
    r1 = r2 - r1;                \
    r2 = r3 - r2;                \
    r3 = r4 - r2;                \
    r4 = r5 - r4;                \
    r5 = r6 - r5;                \
    r6 = r7 - r6;                \
    r7 = r8 - r7;                \
} while (0)

static inline __attribute__((always_inline))
vftype compute_div_vtile(const ftype *__restrict__ src_x,
                         const ftype *__restrict__ src_y,
                         const ftype *__restrict__ src_z,
                         uint32_t height,
                         uint32_t width,
                         vftype prev_rx0,
                         vftype prev_ry0,
                         vftype prev_rz0,
                         int is_first_tile,
                         int is_first_row,
                         int is_first_face,
                         uint32_t dst_stride,
                         ftype *__restrict__ dst)
{
    vftype ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8;
    ry0 = prev_ry0;
    load_vtile(src_y, width, ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8);
    if (is_first_row) {
        ry0 = 2 * (ry1 - ry0);
    } else {
        ry0 = ry1 - ry0;
    }
    fin_diff(ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8);

    vftype rz0, rz1, rz2, rz3, rz4, rz5, rz6, rz7, rz8;
    load_vtile(src_z, width, rz1, rz2, rz3, rz4, rz5, rz6, rz7, rz8);
    if (is_first_face) {
        rz0 = 2 * (rz1 - prev_rz0);
        rz1 = 2 * (rz2 - prev_rz0);
        rz2 = 2 * (rz3 - prev_rz0);
        rz3 = 2 * (rz4 - prev_rz0);
        rz4 = 2 * (rz5 - prev_rz0);
        rz5 = 2 * (rz6 - prev_rz0);
        rz6 = 2 * (rz7 - prev_rz0);
        rz7 = 2 * (rz8 - prev_rz0);
    } else {
        vftype rz1_p, rz2_p, rz3_p, rz4_p, rz5_p, rz6_p, rz7_p, rz8_p;
        load_vtile(src_z - height * width, width,
                   rz1_p, rz2_p, rz3_p, rz4_p, rz5_p, rz6_p, rz7_p, rz8_p);

        rz0 = rz1 - rz1_p;
        rz1 = rz2 - rz2_p;
        rz2 = rz3 - rz3_p;
        rz3 = rz4 - rz4_p;
        rz4 = rz5 - rz5_p;
        rz5 = rz6 - rz6_p;
        rz6 = rz7 - rz7_p;
        rz7 = rz8 - rz8_p;
    }

    rz0 = ry0 + rz0;
    rz1 = ry1 + rz1;
    rz2 = ry2 + rz2;
    rz3 = ry3 + rz3;
    rz4 = ry4 + rz4;
    rz5 = ry5 + rz5;
    rz6 = ry6 + rz6;
    rz7 = ry7 + rz7;

    vtranspose(&rz0, &rz1, &rz2, &rz3, &rz4, &rz5, &rz6, &rz7);

    vftype rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8;
    rx0 = prev_rx0;
    load_vtile(src_x, width, rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8);
    vtranspose(&rx1, &rx2, &rx3, &rx4, &rx5, &rx6, &rx7, &rx8);
    if (is_first_tile) {
        rx0 = 2 * (rx1 - rx0);
    } else {
        rx0 = rx1 - rx0;
    }
    fin_diff(rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8);

    rz0 = rx0 + rz0;
    rz1 = rx1 + rz1;
    rz2 = rx2 + rz2;
    rz3 = rx3 + rz3;
    rz4 = rx4 + rz4;
    rz5 = rx5 + rz5;
    rz6 = rx6 + rz6;
    rz7 = rx7 + rz7;

    vstore(dst + 0 * dst_stride, rz0);
    vstore(dst + 1 * dst_stride, rz1);
    vstore(dst + 2 * dst_stride, rz2);
    vstore(dst + 3 * dst_stride, rz3);
    vstore(dst + 4 * dst_stride, rz4);
    vstore(dst + 5 * dst_stride, rz5);
    vstore(dst + 6 * dst_stride, rz6);
    vstore(dst + 7 * dst_stride, rz7);

    return rx8;
}

static inline __attribute__((always_inline))
void gauss_reduce_vcol(const ftype *__restrict__ f_src,
                       vftype *__restrict__ f_prev,
                       vftype *__restrict__ upper_prev,
                       ftype *__restrict__ upper,
                       ftype *__restrict__ f_dst)
{
    vftype f = vload(f_src);
    vftype norm_coeff = 3 - *upper_prev;

    *upper_prev = 1 / norm_coeff;
    *f_prev = (f - *f_prev) / norm_coeff;

    vstore(upper, *upper_prev);
    vstore(f_dst, *f_prev);
}

static inline __attribute__((always_inline))
void backward_sub_vcol(const ftype *__restrict__ f,
                       const ftype *__restrict__ upper,
                       vftype *__restrict__ p_prev,
                       ftype *__restrict__ p)
{
    vftype f_ = vload(f);
    vftype upp = vload(upper);
    *p_prev = vfmadd(vneg(upp), *p_prev, f_);
    vstore(p, *p_prev);
}

static inline __attribute__((always_inline))
void solve_vtiles_row(const ftype *__restrict__ u_x,
                      const ftype *__restrict__ u_y,
                      const ftype *__restrict__ u_z,
                      uint32_t height,
                      uint32_t width,
                      ftype u0_x,
                      ftype u0_y,
                      ftype u0_z,
                      int is_first_row,
                      int is_first_face,
                      ftype *__restrict__ tmp,
                      ftype *__restrict__ p)
{
    ftype *__restrict__ tmp_upp = tmp;
    ftype *__restrict__ tmp_f = tmp + width * VLEN;

    ftype __attribute__((aligned(32))) div_u_t[VLEN * VLEN];

    vftype last_u_x =
        compute_div_vtile(u_x, u_y, u_z,
                          height, width,
                          vbroadcast(u0_x),
                          is_first_row ? vbroadcast(u0_y) :
                                         vload(u_y - width),
                          /* I can still pass u0_z even if not used. */
                          vbroadcast(u0_z),
                          1, is_first_row, is_first_face,
                          VLEN, div_u_t);

    vftype upp_prev = vbroadcast(0);
    vftype f_prev = vbroadcast(0);
     /* TODO: Apply left BCs. */
    for (int k = 0; k < VLEN; ++k) {
        gauss_reduce_vcol(div_u_t + VLEN * k,
                          &f_prev, &upp_prev,
                          tmp_upp + VLEN * k,
                          tmp_f + VLEN * k);
    }

    for (uint32_t tk = VLEN; tk < width; tk += VLEN) {
        last_u_x = compute_div_vtile(u_x + tk, u_y + tk, u_z + tk,
                                     height, width,
                                     last_u_x,
                                     is_first_row ?
                                        vbroadcast(u0_y) :
                                        vload(u_y - width + tk),
                                     vbroadcast(u0_z),
                                     0, is_first_row, is_first_face,
                                     VLEN, div_u_t);

        for (uint32_t k = 0; k < VLEN; ++k) {
            gauss_reduce_vcol(div_u_t + VLEN * k,
                              &f_prev, &upp_prev,
                              tmp_upp + VLEN * (tk + k),
                              tmp_f + VLEN * (tk + k));
        }
    }

    ftype __attribute__((aligned(32))) p_t[VLEN * VLEN];

    for (uint32_t tk = 0; tk < width; tk += VLEN) {
        vftype p_prev = vbroadcast(0);
        for (uint32_t k = 0; k < VLEN; ++k) {
            backward_sub_vcol(tmp_f + VLEN * (width - 1 - (tk + k)),
                              tmp_upp + VLEN * (width - 1 - (tk + k)),
                              &p_prev,
                              p_t + VLEN * (VLEN - 1 - k));
        }
        transpose_vtile(p_t, VLEN, width, p + width - VLEN - tk);
    }
}

void solve_Dxx_tridiag_blocks(uint32_t depth,
                              uint32_t height,
                              uint32_t width,
                              ftype u0_x,
                              ftype u0_y,
                              ftype u0_z,
                              ftype *__restrict__ tmp,
                              ftype *__restrict__ u_x,
                              ftype *__restrict__ u_y,
                              ftype *__restrict__ u_z,
                              ftype *__restrict__ p)
{
    /* Solve first tile row of the first face. */
    solve_vtiles_row(u_x, u_y, u_z, height, width,
                     u0_x, u0_y, u0_z, 1, 1, tmp, p);
    /* Solve remaining tile rows of the first face. */
    for (uint32_t j = VLEN; j < height; j += VLEN) {
        solve_vtiles_row(u_x + width * j,
                         u_y + width * j,
                         u_z + width * j,
                         height, width,
                         u0_x, u0_y, u0_z,
                         0, 1, tmp,
                         p + width * j);
    }
    /* Solve remaining faces. */
    for (uint32_t i = 1; i < depth; ++i) {
        /* Solve first tile row of the face. */
        solve_vtiles_row(u_x + height * width * i,
                         u_y + height * width * i,
                         u_z + height * width * i,
                         height, width,
                         u0_x, u0_y, u0_z,
                         1, 0, tmp,
                         p + height * width * i);
        /* Solve remaining tile rows of the face. */
        for (uint32_t j = VLEN; j < height; j += VLEN) {
            uint64_t offset = height * width * i + width * j;
            solve_vtiles_row(u_x + offset,
                             u_y + offset,
                             u_z + offset,
                             height, width,
                             u0_x, u0_y, u0_z,
                             0, 0, tmp,
                             p + offset);
        }
    }
}
