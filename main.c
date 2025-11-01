#include "finite-diff.h"
#include "lin-solver.h"

#define DT 0.01f
#define NU 0.01f

static inline __attribute__((always_inline))
vftype compute_g_comp_at(const ftype *__restrict__ eta,
                         const ftype *__restrict__ zeta,
                         uint64_t idx,
                         uint32_t height,
                         uint32_t width,
                         vftype k,
                         vftype u,
                         vftype Dx_pp)
{
    /* Compute second derivatives. */
    /* WARNING: Allocate extra cache line at the start. */
    vftype Dxx_eta = compute_Dxx_at(eta, idx);
    vftype Dyy_zeta = compute_Dyy_at(zeta, idx, width);
    vftype Dzz_u = compute_Dzz_at(u, idx, height * width);

    /* WARNING: Null volume force. */
    vftype f_ = vbroadcast(0.0);
    vftype nu_half = vbroadcast(NU / 2);

    /* yes, it's ugly :/ */
    return vadd(f_,
                vsub(vmul(nu_half,
                          vsub(vadd(Dxx_eta,
                                    vadd(Dyy_zeta,
                                         Dzz_u)),
                               vdiv(u, k))),
                     Dx_pp));
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
    vftype u_ = vload(u + idx);
    vftype g = compute_g_comp_at(
        eta, zeta, idx, height, width, k, u_, D_pp);
     /* TODO: You could avoid this load by using the one
     * needed for the Dxx computation. */
    vftype eta_ = vload(eta + idx);
    return vsub(vadd(u_, vmul(dt_beta, g)), eta_));
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

                /* Compute the gradient of the
                 * pressure predictor pp = p + phi. */
                vftype Dx_p, Dy_p, Dz_p;
                vftype Dx_phi, Dy_phi, Dz_phi;
                /* WARNING: Pressure gradient in the last cell must be 0! */
                compute_grad_at(
                    p, idx, height, width, &Dx_p, &Dy_p, &Dz_p);
                compute_grad_at(
                    phi, idx, height, width, &Dx_phi, &Dy_phi, &Dz_phi);
                vftype Dx_pp = vadd(Dx_p, Dx_pc);
                vftype Dy_pp = vadd(Dy_p, Dy_pc);
                vftype Dz_pp = vadd(Dz_p, Dz_pc);

                vftype k_ = vload(k + idx);
                /* Computes dt / beta */
                vftype dt = vbroadcast(DT);
                vftype dt_nu = vbroadcast(DT * NU);
                vftype kk = vadd(k_, k_);
                vftype dt_beta = vdiv(vmul(kk, dt), vadd(kk, dt_nu));

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
                    uint32_t depth,
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
    uint64_t num_points = depth * heigth * width;
    ftype *__restrict__ Dxx_rhs_x = tmp;
    ftype *__restrict__ Dxx_rhs_y = tmp + num_points;
    ftype *__restrict__ Dxx_rhs_z = tmp + num_points * 2;

    compute_wDxx_rhs(k,
                     depth, height, width,
                     p, phi,
                     eta_x, eta_y, eta_z,
                     zeta_x, zeta_y, zeta_z,
                     u_x, u_y, u_z,
                     Dxx_rhs_x, Dxx_rhs_y, Dxx_rhs_z);

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

int main(void)
{
    /* step(); */

    return 0;
}
