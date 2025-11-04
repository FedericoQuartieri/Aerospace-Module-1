#include "equations.h"
#include "timeit.h"

void step(void)
{
    /* solve_momentum(); */
    /* correct_pressure(); */
}

#include <stdlib.h>

#define D 32
#define H 128
#define W 128

static void rand_fill(ftype *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

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
