#include "finite-diff.h"
#include "lin-solver.h"
#include "timeit.h"

#define DT 0.01f
#define NU 0.01f

static inline __attribute__((always_inline))
vftype compute_g_comp_at(const ftype *__restrict__ eta,
                         const ftype *__restrict__ zeta,
                         const ftype *__restrict__ u,
                         uint64_t idx,
                         uint32_t height,
                         uint32_t width,
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
    vftype nu_half = vbroadcast(NU / 2);

    /* yes, it's ugly :/ */
    return vadd(f_,
                vsub(vmul(nu_half,
                          vsub(vadd(Dxx_eta,
                                    vadd(Dyy_zeta,
                                         Dzz_u)),
                               vdiv(u_, k))),
                     D_pp));
}

static inline __attribute__((always_inline))
vftype compute_wDxx_rhs_comp_at(const ftype *__restrict__ eta,
                                const ftype *__restrict__ zeta,
                                const ftype *__restrict__ u,
                                uint64_t idx,
                                uint32_t height,
                                uint32_t width,
                                vftype k,
                                vftype D_pp,
                                vftype dt_beta)
{
    vftype eta_ = vload(eta + idx);
    vftype zeta_ = vload(zeta + idx);
    vftype u_ = vload(u + idx);
    vftype g = compute_g_comp_at(
        eta, zeta, u, idx, height, width, k, eta_, zeta_, u_, D_pp);
    return vsub(vadd(u_, vmul(dt_beta, g)), eta_);
}

static void compute_wDxx_rhs(
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
    ftype *__restrict__ rhs_x,
    ftype *__restrict__ rhs_y,
    ftype *__restrict__ rhs_z)
{

    vftype zeros = vbroadcast(0.0);
    vftype dt = vbroadcast(DT);
    vftype dt_nu = vbroadcast(DT * NU);

    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
#ifdef AUTO_VEC
            for (uint32_t l = 0; l < width; ++l) {
                /* TODO: Scalar version. */
            }
#else
            for (uint32_t l = 0; l < width; l += VLEN) {
                uint64_t idx = height * width * i + width * j + l;

                /* TODO: Consider on the fly rhs evaluation while solving. */

                /* Compute the gradient of pressure predictor pp = p + phi. */
                vftype Dx_p, Dy_p, Dz_p;
                vftype Dx_phi, Dy_phi, Dz_phi;
                compute_grad_at(
                    p, idx, height, width, &Dx_p, &Dy_p, &Dz_p);
                compute_grad_at(
                    phi, idx, height, width, &Dx_phi, &Dy_phi, &Dz_phi);
                vftype Dx_pp = vadd(Dx_p, Dx_phi);
                /* WARNING: Pressure gradient in the last cell must be 0!
                  These conditionals don't hurt performance apparently */
                vftype Dy_pp = (j == height - 1) ? zeros : vadd(Dy_p, Dy_phi);
                vftype Dz_pp = (i == depth - 1) ? zeros : vadd(Dz_p, Dz_phi);

                /* Computes dt/beta */
                vftype k_ = vload(k + idx);
                vftype kk = vadd(k_, k_);
                vftype dt_beta = vdiv(vmul(kk, dt), vadd(kk, dt_nu));

                /* NT stores make no difference here,
                 * loads are bottlenecking. */
                vstore(rhs_x + idx,
                       compute_wDxx_rhs_comp_at(eta_x, zeta_x, u_x,
                                                idx, height, width,
                                                k_, Dx_pp, dt_beta));
                vstore(rhs_y + idx,
                       compute_wDxx_rhs_comp_at(eta_y, zeta_y, u_y,
                                                idx, height, width,
                                                k_, Dy_pp, dt_beta));
                vstore(rhs_z + idx,
                       compute_wDxx_rhs_comp_at(eta_z, zeta_z, u_z,
                                                idx, height, width,
                                                k_, Dz_pp, dt_beta));
            }
#endif
        }
    }
}


void solve_momentum(const ftype *__restrict__ k, /* Porosity. */
                    uint32_t depth,
                    uint32_t height,
                    uint32_t width,
                    ftype *__restrict__ tmp,
                    /* Pressure from previous half-step. */
                    ftype *__restrict__ p,
                    /* Pressure correction from previous half-step. */
                    ftype *__restrict__ phi,
                    /* (I - wDxx) velocity from previous step */
                    ftype *__restrict__ eta_x,
                    ftype *__restrict__ eta_y,
                    ftype *__restrict__ eta_z,
                    /* (I - wDyy) velocity from previous step */
                    ftype *__restrict__ zeta_x,
                    ftype *__restrict__ zeta_y,
                    ftype *__restrict__ zeta_z,
                    /* Velocity from previous step */
                    ftype *__restrict__ u_x,
                    ftype *__restrict__ u_y,
                    ftype *__restrict__ u_z)
{
    uint64_t num_points = depth * height * width;
    ftype *__restrict__ Dxx_rhs_x = tmp;
    ftype *__restrict__ Dxx_rhs_y = tmp + num_points;
    ftype *__restrict__ Dxx_rhs_z = tmp + num_points * 2;

    TIMEIT(compute_wDxx_rhs(k,
                            p, phi,
                            eta_x, eta_y, eta_z,
                            zeta_x, zeta_y, zeta_z,
                            u_x, u_y, u_z,
                            depth, height, width,
                            Dxx_rhs_x, Dxx_rhs_y, Dxx_rhs_z));

    /* TODO: What if we alternate solving a group of Dxx rows,
     * and advancing the Dyy solver reduction over those solved rows,
     * then proceed until the whole face has been solved for Dyy?
     * The same reasoning could apply to Dzz. So essentially we would
     * do a single sweep over the full domain, fusing the Dxx, Dyy and
     * Dzz routines, instead of three separate passes for each solver.
     * I guess the number of iterations would be the same, but we're
     * would maximize the temporal reuse of the data.
     * I think it's worth investigating. */

    //solve_wDxx_tridiag_blocks();
    //solve_wDyy_tridiag_blocks();
    //solve_wDzz_tridiag_blocks();

}

void solve_pressure(void)
{

}

void step(void)
{
    /* solve_momentum(); */
    /* correct_pressure(); */
}

#include <stdlib.h>

#define D 256
#define H 256
#define W 256

int main(void)
{
    /* step(); */
    size_t size = (D + 2) * H * W;
    ftype *k = aligned_alloc(32, size * sizeof(ftype));
    ftype *p = aligned_alloc(32, size * sizeof(ftype));
    ftype *phi = aligned_alloc(32, size * sizeof(ftype));

    ftype *eta_x = aligned_alloc(32, size * sizeof(ftype));
    ftype *eta_y = aligned_alloc(32, size * sizeof(ftype));
    ftype *eta_z = aligned_alloc(32, size * sizeof(ftype));

    ftype *zeta_x = aligned_alloc(32, size * sizeof(ftype));
    ftype *zeta_y = aligned_alloc(32, size * sizeof(ftype));
    ftype *zeta_z = aligned_alloc(32, size * sizeof(ftype));

    ftype *u_x = aligned_alloc(32, size * sizeof(ftype));
    ftype *u_y = aligned_alloc(32, size * sizeof(ftype));
    ftype *u_z = aligned_alloc(32, size * sizeof(ftype));

    ftype *tmp = aligned_alloc(32, size * sizeof(ftype) * 3);

    solve_momentum(k,
                   D, H, W,
                   tmp,
                   p, phi,
                   eta_x + W * H, eta_y + H * W, eta_z + H * W,
                   zeta_x + H * W, zeta_y + H * W, zeta_z + H * W,
                   u_x + H * W, u_y + H * W, u_z + H * W);

    free(k);
    free(p);
    free(phi);
    free(eta_x);
    free(eta_y);
    free(eta_z);
    free(zeta_x);
    free(zeta_y);
    free(zeta_z);
    free(u_x);
    free(u_y);
    free(u_z);
    free(tmp);

    return 0;
}
