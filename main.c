#include "equations.h"
#include "timeit.h"
#include "utils.h"

void step(void)
{
    /* solve_momentum(); */
    /* correct_pressure(); */
}

#include <stdlib.h>

#define D 32
#define H 128
#define W 128

#define NU 1.0
#define DT 0.01
#define DX 0.01

int main(void)
{
    /* step(); */
    size_t size = (D + 2) * H * W;
    ftype *k = aligned_alloc(32, size * sizeof(ftype));
    /* TODO: avoid allocating w, compute it on the fly from k. */
    ftype *w = aligned_alloc(32, size * sizeof(ftype));
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

    ftype *tmp = aligned_alloc(32, size * sizeof(ftype) * 6);

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

    /* Precomputing w, consider computing it on the fly when solving. */
    /* WARNING: Scale w by 1/dx^2 */
    compute_w(k, D, H, W, NU, DT, w);

    solve_momentum(k,
                   w,
                   D, H, W,
                   NU, DT, DX,
                   tmp,
                   p, phi,
                   eta_x + W * H, eta_y + H * W, eta_z + H * W,
                   zeta_x + H * W, zeta_y + H * W, zeta_z + H * W,
                   u_x + H * W, u_y + H * W, u_z + H * W);

    free(k);
    free(w);
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
