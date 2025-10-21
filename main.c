#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ftype.h"
#include "timeit.h"
#include "lin-solver.h"
#include "finite-diff.h"

#define D 128
#define H 1024
#define W (1024 + 128) /* Avoid cache aliasing. */

#define TD 1
#define TH 128
#define TW 128

static void rand_fill(ftype *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((ftype) rand()) / RAND_MAX;
    }
}

int main(void)
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

    ftype *grad_i = aligned_alloc(32, size * sizeof(ftype));
    ftype *grad_j = aligned_alloc(32, size * sizeof(ftype));
    ftype *grad_k = aligned_alloc(32, size * sizeof(ftype));
    ftype *grad_i_tiled = aligned_alloc(32, size * sizeof(ftype));
    ftype *grad_j_tiled = aligned_alloc(32, size * sizeof(ftype));
    ftype *grad_k_tiled = aligned_alloc(32, size * sizeof(ftype));

    TIMEITN(compute_grad(w, D, H, W, grad_i, grad_j, grad_k), 50);
    TIMEITN(compute_grad_strided(
        w, D, H, W, grad_i_tiled, grad_j_tiled, grad_k_tiled), 50);
    TIMEITN(compute_grad_tiled(
        w, D, H, W, TD, TH, TW, grad_i_tiled, grad_j_tiled, grad_k_tiled), 50);

    assert(memcmp(grad_i, grad_i_tiled, size * sizeof(ftype)) == 0);
    assert(memcmp(grad_j, grad_j_tiled, size * sizeof(ftype)) == 0);
    assert(memcmp(grad_k, grad_k_tiled, size * sizeof(ftype)) == 0);

    free(grad_k_tiled);
    free(grad_j_tiled);
    free(grad_i_tiled);
    free(grad_k);
    free(grad_j);
    free(grad_i);

    free(tmp);
    free(f);
    free(u);
    free(w);

    return 0;
}
