#include <stdio.h>

#include "lin-solver.h"
#include "boundary.h"
#include "consts.h"
#include "finite-diff.h"

static inline __attribute__((always_inline))
vftype compute_g_comp_at(const ftype *__restrict__ eta,
                         const ftype *__restrict__ zeta,
                         const ftype *__restrict__ u,
                         uint64_t idx,
                         uint32_t height,
                         uint32_t width,
                         ftype nu,
                         ftype dx,
                         vftype k,
                         vftype eta_,
                         vftype zeta_,
                         vftype u_,
                         vftype D_pp)
{
    /* Compute second derivatives. */
    /* WARNING: Allocate extra face at the start. */
    vftype Dxx_eta = compute_Dxx_at(eta, idx, eta_);
    vftype Dyy_zeta = compute_Dyy_at(zeta, idx, width, zeta_);
    vftype Dzz_u = compute_Dzz_at(u, idx, height * width, u_);

    /* WARNING: Null volume force. */
    vftype f_ = vbroadcast(0.0);
    vftype nu_half = vbroadcast(nu / 2);
    vftype inv_dx = vbroadcast(1 / dx);
    vftype inv_dxdx = vbroadcast(1 / (dx * dx));

    /* yes, it's ugly :/ */
    return vadd(f_,
                vsub(vmul(nu_half,
                          vsub(vmul(vadd(Dxx_eta,
                                         vadd(Dyy_zeta,
                                              Dzz_u)),
                                    inv_dxdx),
                               vdiv(u_, k))),
                     vmul(D_pp,
                          inv_dx)));
}

static inline __attribute__((always_inline))
vftype compute_momentum_Dxx_rhs_comp_at(const ftype *__restrict__ eta,
                                        const ftype *__restrict__ zeta,
                                        const ftype *__restrict__ u,
                                        uint64_t idx,
                                        uint32_t height,
                                        uint32_t width,
                                        ftype nu,
                                        ftype dx,
                                        vftype k,
                                        vftype D_pp,
                                        vftype dt_beta)
{
    vftype eta_ = vload(eta + idx);
    vftype zeta_ = vload(zeta + idx);
    vftype u_ = vload(u + idx);
    vftype g = compute_g_comp_at(eta, zeta, u,
                                 idx, height, width, nu, dx,
                                 k, eta_, zeta_, u_, D_pp);
    return vsub(vfmadd(dt_beta, g, u_), eta_);
}

void compute_momentum_Dxx_rhs(
    const ftype *__restrict__ k, /* Porosity. */
    /* Pressure from previous half-step. */
    const ftype *__restrict__ p,
    /* Pressure correction from previous half-step. */
    const ftype *__restrict__ phi,
    /* (I - wDxx) velocity from previous step */
    const ftype *__restrict__ eta_x,
    const ftype *__restrict__ eta_y,
    const ftype *__restrict__ eta_z,
    /* (I - wDyy) velocity from previous step */
    const ftype *__restrict__ zeta_x,
    const ftype *__restrict__ zeta_y,
    const ftype *__restrict__ zeta_z,
    /* Velocity from previous step */
    const ftype *__restrict__ u_x,
    const ftype *__restrict__ u_y,
    const ftype *__restrict__ u_z,
    uint32_t depth,
    uint32_t height,
    uint32_t width,
    ftype u_ex_x,
    ftype u_ex_y,
    ftype u_ex_z,
    ftype nu, /* Viscosity */
    ftype dt, /* Timestep size */
    ftype dx, /* Grid cell size */
    ftype *__restrict__ rhs_x,
    ftype *__restrict__ rhs_y,
    ftype *__restrict__ rhs_z)
{
    /* TODO: Consider on the fly rhs evaluation while solving. */

    vftype vdt = vbroadcast(dt);
    vftype vdt_nu = vbroadcast(dt * nu);

    /* We can avoid computing rhs at i = 0, j = 0, k = 0,
     * i = depth - 1 (for z), j = height - 1 (for y)
     * and k = width - 1 (for x). */

    for (uint32_t i = 1; i < depth; ++i) {
        for (uint32_t j = 1; j < height; j++) {
            for (uint32_t l = 0; l < width; l += VLEN) {
                uint64_t idx = height * width * i + width * j + l;

                /* WARNING: Not too sure about were not to compute rhs. */

                /* Compute the gradient of pressure predictor pp = p + phi. */
                vftype Dx_p, Dy_p, Dz_p;
                compute_grad_at(p, idx, height, width,
                                &Dx_p, &Dy_p, &Dz_p);
                vftype Dx_phi, Dy_phi, Dz_phi;
                compute_grad_at(phi, idx, height, width,
                                &Dx_phi, &Dy_phi, &Dz_phi);

                vftype Dx_pp = vadd(Dx_p, Dx_phi);
                vftype Dy_pp = vadd(Dy_p, Dy_phi);
                vftype Dz_pp = vadd(Dz_p, Dz_phi);

                /* Computes dt/beta */
                vftype k_ = vload(k + idx);
                vftype kk = vadd(k_, k_);
                vftype dt_beta = vdiv(vmul(kk, vdt), vadd(kk, vdt_nu));

                /* NT stores make no difference here,
                 * loads are bottlenecking. */
                vstore(rhs_x + idx,
                       compute_momentum_Dxx_rhs_comp_at(eta_x, zeta_x, u_x,
                                                        idx, height, width,
                                                        nu, dx, k_, Dx_pp,
                                                        dt_beta));
                vstore(rhs_y + idx,
                       compute_momentum_Dxx_rhs_comp_at(eta_y, zeta_y, u_y,
                                                        idx, height, width,
                                                        nu, dx, k_, Dy_pp,
                                                        dt_beta));
                vstore(rhs_z + idx,
                       compute_momentum_Dxx_rhs_comp_at(eta_z, zeta_z, u_z,
                                                        idx, height, width,
                                                        nu, dx, k_, Dz_pp,
                                                        dt_beta));
            }

            /* For y and z, correct Dxx(eta) using the ghost node. */
            uint64_t idx = height * width * i + width * j + width - 1;
            ftype coeff = 2 * k[idx] * dt / (2 * k[idx] + dt * nu) *
                          nu / (2 * dx * dx);
            rhs_y[idx] -= coeff * (eta_y[idx + 1] + eta_y[idx] - 2 * u_ex_y);
            rhs_z[idx] -= coeff * (eta_z[idx + 1] + eta_z[idx] - 2 * u_ex_z);
        }

        /* TODO: Temporary patch, perform loop peeling instead. */
        for (uint32_t l = 0; l < width; l += VLEN) {
            uint64_t idx = height * width * i + width * (height - 1) + l;
            /* For x and z, correct Dyy(zeta) using the ghost node. */

            vftype k_ = vload(k + idx);
            vftype coeff = 2 * k_ * dt / (2 * k_ + dt * nu) *
                           nu / (2 * dx * dx);

            vstore(rhs_x + idx, vload(rhs_x + idx) -
                                coeff * (vload(zeta_x + idx + width) +
                                         vload(zeta_x + idx) - 2 * u_ex_x));

            vstore(rhs_z + idx, vload(rhs_z + idx) -
                                coeff * (vload(zeta_z + idx + width) +
                                         vload(zeta_z + idx) - 2 * u_ex_z));
        }
    }

    /* TODO: Temporary patch, perform loop peeling instead. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t l = 0; l < width; l += VLEN) {
            uint64_t idx = height * width * (depth - 1) + width * j + l;
            /* For x and y, correct Dzz(u) using the ghost node. */

            vftype k_ = vload(k + idx);
            vftype coeff = 2 * k_ * dt / (2 * k_ + dt * nu) *
                           nu / (2 * dx * dx);

            vstore(rhs_x + idx, vload(rhs_x + idx) -
                                coeff * (vload(u_x + idx + height * width) +
                                         vload(u_x + idx) - 2 * u_ex_x));

            vstore(rhs_y + idx, vload(rhs_y + idx) -
                                coeff * (vload(u_y + idx + height * width) +
                                         vload(u_y + idx) - 2 * u_ex_y));
        }
    }
}

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
static void solve_momentum_Dxx(const ftype *__restrict__ w,
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

    /* TODO: Why don't you write the solution directly to f?
     * You can reduce the number of cache misses. */

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
static void solve_momentum_Dyy(const ftype *__restrict__ w,
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
static void solve_momentum_Dzz(const ftype *__restrict__ w,
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


void momentum_init(field_size size, field3 field);
{
    uint64_t size = size.depth * size.height * size.width * sizeof(ftype);

    memset(field.x, 0, size);
    memset(field.y, 0, size);
    memset(field.z, 0, size);

    /* Initialize front face. */
    for (uint32_t j = 0; j < size.height; ++j) {
        for (uint32_t k = 0; k < size.width; k += VLEN) {
            vftype u_x, u_y, u_z;
            _get_front_bc_u(k, j, 0, &u_x, &u_y, &u_z);

            uint64_t idx = width * j + k;
            vstore(field.x + idx, u_x);
            vstore(field.y + idx, u_y);
            vstore(field.z + idx, u_z);
    }

    for (uint32_t i = 1; i < size.depth - 1; ++i) {
        /* Initialize top face. */
        for (uint32_t k = 0; k < size.width; k += VLEN) {
            vftype u_x, u_y, u_z;
            _get_top_bc_u(k, 0, i, &u_x, &u_y, &u_z);

            uint64_t idx = size.height * size.width * i + k;
            vstore(field.x + idx, u_x);
            vstore(field.y + idx, u_y);
            vstore(field.z + idx, u_z);
        }

        for (uint32_t j = 1; j < size.height - 1; ++j) {
            /* TODO: I need a scalar bc getter. */
        }

        /* Initialize bottom face. */
        for (uint32_t k = 0; k < size.width; k += VLEN) {
            vftype u_x, u_y, u_z;
            _get_bottom_bc_u(k, size.height - 1, i, &u_x, &u_y, &u_z);

            uint64_t idx = size.height * size.width * i +
                           (size.height - 1) * j + k;
            vstore(field.x + idx, u_x);
            vstore(field.y + idx, u_y);
            vstore(field.z + idx, u_z);
        }
    }

    /* Initialize back face. */
    for (uint32_t j = 0; j < size.height; ++j) {
        for (uint32_t k = 0; k < size.width; k += VLEN) {
            vftype u_x, u_y, u_z;
            _get_back_bc_u(k, j, 0, &u_x, &u_y, &u_z);

            uint64_t idx = size.height * size.width *
                           (depth - 1) + size.width * j + k;
            vstore(field.x + idx, u_x);
            vstore(field.y + idx, u_y);
            vstore(field.z + idx, u_z);
    }
}

void momentum_solve()
{
    return;
}
