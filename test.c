#include <stdio.h>
#include <stdlib.h>

#include "lin-solver.h"
#include "ftype.h"

#define D 16
#define H 16
#define W 16

int main(void)
{
    srand(42);

    size_t size = (D + VLEN) * (H + VLEN) * (W + VLEN);
    ftype *w = aligned_alloc(32, size * sizeof(ftype));
    ftype *u = aligned_alloc(32, size * sizeof(ftype));
    ftype *f = aligned_alloc(32, size * sizeof(ftype));
    ftype *tmp = aligned_alloc(32, size * sizeof(ftype));

    for (int i = 0; i < D * H * W; ++i) {
        w[i] = 1;//((ftype) rand()) / RAND_MAX;
        f[i] = 1;//((ftype) rand()) / RAND_MAX;
    }

    solve_wDxx_tridiag_blocks(w, D, H, W, tmp, f, u);

    for (int i = 0; i < H * W; ++i) {
        printf("%f\n", u[i]);
    }

    free(tmp);
    free(f);
    free(u);
    free(w);

    return 0;
}
