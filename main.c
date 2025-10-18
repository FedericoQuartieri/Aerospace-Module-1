#include <stdlib.h>

#include "ftype.h"
#include "timeit.h"
#include "lin-solver.h"

#define D 128
#define H 512
#define W 544 /* Avoid cache aliasing. */

static void rand_fill(ftype *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

int main(void)
{
    size_t size = D * H * W;
    ftype *w = malloc(size * sizeof(ftype));
    ftype *u = malloc(size * sizeof(ftype));
    ftype *f = malloc(size * sizeof(ftype));
    ftype *tmp = malloc(size * sizeof(ftype));

    rand_fill(w, size);
    rand_fill(f, size);

    TIMEIT(solve_wDxx_tridiag_blocks(w, D, H, W, tmp, f, u));
    TIMEIT(solve_wDyy_tridiag_blocks(w, D, H, W, tmp, f, u));
    TIMEIT(solve_wDzz_tridiag_blocks(w, D, H, W, tmp, f, u));

    return 0;
}
