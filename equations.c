#include "equations.h"
#include "finite-diff.h"

void compute_w(const ftype *__restrict__ k,
               uint32_t depth,
               uint32_t height,
               uint32_t width,
               ftype nu,
               ftype dt,
               ftype *__restrict__ w)
{
    /* WARNING: Scale w by 1/dx^2 */
    vftype dt_nu = vbroadcast(nu * dt);
    for (uint64_t i = 0; i < depth * height * width; i += VLEN) {
        vftype k_ = vload(k + i);
        /* w = (k dt nu) / (2k + dt nu) */
        vstore(w, vdiv(vmul(k_, dt_nu), vadd(vadd(k_, k_), dt_nu)));
    }
}

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
vftype compute_wDxx_rhs_comp_at(const ftype *__restrict__ eta,
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

void compute_wDxx_rhs(
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
    ftype nu, /* Viscosity */
    ftype dt, /* Timestep size */
    ftype dx, /* Grid cell size */
    ftype *__restrict__ rhs_x,
    ftype *__restrict__ rhs_y,
    ftype *__restrict__ rhs_z)
{
    /* TODO: Consider on the fly rhs evaluation while solving. */

    vftype vzero = vbroadcast(0.0);
    vftype vdt = vbroadcast(dt);
    vftype vdt_nu = vbroadcast(dt * nu);

    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; j++) {
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
                /* Pressure gradient in the last cell must be 0!
                   These conditionals don't hurt performance apparently */
                vftype Dy_pp = (j == height - 1) ? vzero : vadd(Dy_p, Dy_phi);
                vftype Dz_pp = (i == depth - 1) ? vzero : vadd(Dz_p, Dz_phi);

                /* Computes dt/beta */
                vftype k_ = vload(k + idx);
                vftype kk = vadd(k_, k_);
                vftype dt_beta = vdiv(vmul(kk, vdt), vadd(kk, vdt_nu));

                /* NT stores make no difference here,
                 * loads are bottlenecking. */
                vstore(rhs_x + idx,
                       compute_wDxx_rhs_comp_at(eta_x, zeta_x, u_x,
                                                idx, height, width, nu, dx,
                                                k_, Dx_pp, dt_beta));
                vstore(rhs_y + idx,
                       compute_wDxx_rhs_comp_at(eta_y, zeta_y, u_y,
                                                idx, height, width, nu, dx,
                                                k_, Dy_pp, dt_beta));
                vstore(rhs_z + idx,
                       compute_wDxx_rhs_comp_at(eta_z, zeta_z, u_z,
                                                idx, height, width, nu, dx,
                                                k_, Dz_pp, dt_beta));
            }
        }
    }
}

void solve_momentum(const ftype *__restrict__ k,
                    const ftype *__restrict__ w,
                    uint32_t depth,
                    uint32_t height,
                    uint32_t width,
                    ftype nu,
                    ftype dt,
                    ftype dx,
                    ftype *__restrict__ tmp,
                    ftype *__restrict__ p,
                    ftype *__restrict__ phi,
                    ftype *__restrict__ eta_x,
                    ftype *__restrict__ eta_y,
                    ftype *__restrict__ eta_z,
                    ftype *__restrict__ zeta_x,
                    ftype *__restrict__ zeta_y,
                    ftype *__restrict__ zeta_z,
                    ftype *__restrict__ u_x,
                    ftype *__restrict__ u_y,
                    ftype *__restrict__ u_z)
{
    uint64_t num_points = depth * height * width;
    /* WARNING: Cache aliasing? */
    ftype *__restrict__ wDxx_rhs_x = tmp;
    ftype *__restrict__ wDxx_rhs_y = tmp + num_points;
    ftype *__restrict__ wDxx_rhs_z = tmp + num_points * 2;

    compute_wDxx_rhs(k, p, phi,
                     eta_x, eta_y, eta_z,
                     zeta_x, zeta_y, zeta_z,
                     u_x, u_y, u_z,
                     depth, height, width,
                     nu, dt, dx,
                     wDxx_rhs_x, wDxx_rhs_y, wDxx_rhs_z);

    /* TODO: What if we alternate solving a group of Dxx rows,
     * and advancing the Dyy solver reduction over those solved rows,
     * then proceed until the whole face has been solved for Dyy?
     * The same reasoning could apply to Dzz. So essentially we would
     * do a single sweep over the full domain, fusing the Dxx, Dyy and
     * Dzz routines, instead of three separate passes for each solver.
     * I guess the number of iterations would be the same, but we're
     * would maximize the temporal reuse of the data.
     * I think it's worth investigating. */


    /* TODO: Let solve_* perform this sum for solved blocks. */

    //solve_wDxx_tridiag_blocks();
    //solve_wDyy_tridiag_blocks();
    //solve_wDzz_tridiag_blocks();
}

void solve_pressure(void)
{

}


