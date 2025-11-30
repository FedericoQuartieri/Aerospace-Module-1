#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ftype.h"
#include "timeit.h"
#include "utils.h"
#include "lin-solver.h"
#include "equations.h"
#include "boundary.h"

#define D 128
#define H 128
#define W 128 /* + something to avoid cache aliasing? */

DEFINE_CONSTANT_BC_U(0, 0, 0, BC_LEFT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_RIGHT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_TOP)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BOTTOM)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_FRONT)
DEFINE_CONSTANT_BC_U(0, 0, 0, BC_BACK)

void benchmark_solvers(void)
{
    size_t size = (D + VLEN) * (H + VLEN) * (W + VLEN);
    ftype *w = aligned_alloc(32, size * sizeof(ftype));
    ftype *u = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *f = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *tmp = aligned_alloc(32, size * sizeof(ftype));
    ftype *p = aligned_alloc(32, size * sizeof(ftype));

    /* WARNING: Cache aliasing? */
    ftype *u_x = u;
    ftype *f_x = f;
    ftype *u_y = u + size;
    ftype *f_y = f + size;
    ftype *u_z = u + 2 * size;
    ftype *f_z = f + 2 * size;

    /* Write something! Otherwise no physical memory
     * is allocated if you're only reading from it. */
    rand_fill(w, size);
    rand_fill(f_x, size);
    rand_fill(f_y, size);
    rand_fill(f_z, size);

    /*
    TIMEIT(solve_wDxx_tridiag_blocks(w, D, H, W,
                                     tmp, f_x, f_y, f_z, u_x, u_y, u_z));
    TIMEIT(solve_wDyy_tridiag_blocks(w, D, H, W,
                                     tmp, f_x, f_y, f_z, u_x, u_y, u_z));
    TIMEIT(solve_wDzz_tridiag_blocks(w, D, H, W,
                                     tmp, f_x, f_y, f_z, u_x, u_y, u_z));
    */

    TIMEIT(solve_pressure_Dxx(D, H, W, tmp, u_x, u_y, u_z, w));
    TIMEIT(solve_pressure_Dyy(D, H, W, tmp, w, p));
    TIMEIT(solve_pressure_Dzz(D, H, W, tmp, p, w));

    TIMEIT(solve_pressure_fused(D, H, W, tmp, u_x, u_y, u_z, p));

    free(p);
    free(tmp);
    free(f);
    free(u);
    free(w);
}

void benchmark_momentum_Dxx_rhs_computation(void)
{
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
    ftype *rhs_x = aligned_alloc(32, size * sizeof(ftype));
    ftype *rhs_y = aligned_alloc(32, size * sizeof(ftype));
    ftype *rhs_z = aligned_alloc(32, size * sizeof(ftype));

    rand_fill(k, size);
    rand_fill(p, size);
    rand_fill(phi, size);
    rand_fill(eta_x, size);
    rand_fill(eta_y, size);
    rand_fill(eta_z, size);
    rand_fill(zeta_x, size);
    rand_fill(zeta_y, size);
    rand_fill(zeta_z, size);
    rand_fill(u_x, size);
    rand_fill(u_y, size);
    rand_fill(u_z, size);

    ftype nu = ((ftype) rand()) / RAND_MAX;
    ftype dt = ((ftype) rand()) / RAND_MAX;
    ftype dx = ((ftype) rand()) / RAND_MAX;

    uint32_t face_size = H * W;

    TIMEIT(compute_momentum_Dxx_rhs(
        k, p, phi,
        eta_x + face_size, eta_y + face_size, eta_z + face_size,
        zeta_x + face_size, zeta_y + face_size, zeta_z + face_size,
        u_x + face_size, u_y + face_size, u_z + face_size,
        D, H, W,
        0.0, 0.0, 0.0,
        nu, dt, dx,
        rhs_x, rhs_y, rhs_z));

    free(rhs_z);
    free(rhs_y);
    free(rhs_x);
    free(u_z);
    free(u_y);
    free(u_x);
    free(zeta_z);
    free(zeta_y);
    free(zeta_x);
    free(eta_z);
    free(eta_y);
    free(eta_x);
    free(phi);
    free(p);
    free(k);
}

int main(void)
{
    benchmark_solvers();
    benchmark_momentum_Dxx_rhs_computation();
    return 0;
}
