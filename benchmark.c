#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ftype.h"
#include "timeit.h"
#include "lin-solver.h"
#include "equations.h"

#define D 256
#define H 256
#define W 256 /* + something to avoid cache aliasing? */

static void rand_fill(ftype *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

void benchmark_wD_solvers(void)
{
    size_t size = (D + VLEN) * (H + VLEN) * (W + VLEN);
    ftype *w = aligned_alloc(32, size * sizeof(ftype));
    ftype *u = aligned_alloc(32, size * sizeof(ftype));
    ftype *f = aligned_alloc(32, size * sizeof(ftype));
    ftype *tmp = aligned_alloc(32, size * sizeof(ftype));

    rand_fill(w, size);
    rand_fill(f, size);

    TIMEIT(solve_wDxx_tridiag_blocks(w, D, H, W, tmp, f, u));
    TIMEIT(solve_wDyy_tridiag_blocks(w, D, H, W, tmp, f, u));
    TIMEIT(solve_wDzz_tridiag_blocks(w, D, H, W, tmp, f, u));

    free(tmp);
    free(f);
    free(u);
    free(w);
}

void benchmark_wDxx_rhs_computation(void)
{
    size_t size = (D + 2) * H * W;

    ftype *k = aligned_alloc(32, size * sizeof(ftype));
    ftype *p = aligned_alloc(32, size * sizeof(ftype));
    ftype *phi = aligned_alloc(32, size * sizeof(ftype));
    ftype *eta = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *zeta = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *u = aligned_alloc(32, 3 * size * sizeof(ftype));
    ftype *rhs = aligned_alloc(32, 3 * size * sizeof(ftype));

    rand_fill(k, size);
    rand_fill(p, size);
    rand_fill(phi, size);
    rand_fill(eta, 3 * size);
    rand_fill(zeta, 3 * size);
    rand_fill(u, 3 * size);

    uint32_t face_size = H * W;
    ftype *eta_x = eta + face_size;
    ftype *eta_y = eta + face_size + size;
    ftype *eta_z = eta + face_size + 2 * size;
    ftype *zeta_x = zeta + face_size;
    ftype *zeta_y = zeta + face_size + size;
    ftype *zeta_z = zeta + face_size + 2 * size;
    ftype *u_x = u + face_size;
    ftype *u_y = u + face_size + size;
    ftype *u_z = u + face_size + 2 * size;
    ftype *rhs_x = rhs;
    ftype *rhs_y = rhs + size;
    ftype *rhs_z = rhs + 2 * size;

    ftype nu = ((ftype) rand()) / RAND_MAX;
    ftype dt = ((ftype) rand()) / RAND_MAX;
    ftype dx = ((ftype) rand()) / RAND_MAX;

    TIMEIT(compute_wDxx_rhs(k, p, phi,
                            eta_x, eta_y, eta_z,
                            zeta_x, zeta_y, zeta_z,
                            u_x, u_y, u_z,
                            D, H, W,
                            nu,
                            dt,
                            dx,
                            rhs_x, rhs_y, rhs_z));

    free(rhs);
    free(u);
    free(zeta);
    free(eta);
    free(phi);
    free(p);
    free(k);
}

int main(void)
{
    benchmark_wD_solvers();
    benchmark_wDxx_rhs_computation();
    return 0;
}
