/*
 * Does the Schur complement give the same answer as plain Thomas?
 *
 * Splitting a tridiagonal solve across processes is the one thing the
 * direction-splitting scheme cannot do naively: the forward sweep needs the
 * previous point, so a second process would just wait for the first.  The
 * Schur complement removes that wait, and it is an exact method, so the
 * answer must match the sequential one to round-off.
 *
 * Here the blocks are still slices of one array in one process, and nothing
 * is communicated.  That is the point: it checks the algebra while it is
 * still debuggable with an ordinary debugger.  Replacing the slices with
 * processes, and the shared values with messages, comes later.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "parallel.h"
#include "schur.h"
#include "utils.h"

/* Reproducible pseudo-random numbers in [0, 1), so a failure can be replayed. */
static unsigned int random_state = 12345u;

static Real next_random(void) {
    random_state = random_state * 1103515245u + 12345u;
    return (Real)((random_state >> 16) & 0x7fffu) / (Real)0x7fffu;
}

/*
 * A diagonally dominant system: Thomas is stable on it without pivoting, and
 * so are the block solves the Schur complement performs.
 */
static void build_system(int n, Real *a, Real *b, Real *c, Real *f) {
    random_state = 12345u;

    for (int i = 0; i < n; i++) {
        a[i] = -(Real)1 - next_random();
        c[i] = -(Real)1 - next_random();
        b[i] = (Real)4 + next_random();
        f[i] = next_random() - (Real)0.5;
    }
}

static Real largest_difference(int n, const Real *p, const Real *q) {
    Real worst = (Real)0;

    for (int i = 0; i < n; i++) {
        Real difference = (Real)fabs((double)(p[i] - q[i]));
        /* Written inverted so that a NaN, whose comparisons are all
         * false, becomes the worst value instead of being skipped. */
        if (!(difference <= worst)) {
            worst = difference;
        }
    }

    return worst;
}

/* Errors this size are the reordered floating-point sums, nothing else. */
#define TOLERANCE ((Real)1e-12)

static int check(int n, int blocks) {
    Real *a = xmalloc((size_t)n * sizeof(Real));
    Real *b = xmalloc((size_t)n * sizeof(Real));
    Real *c = xmalloc((size_t)n * sizeof(Real));
    Real *f = xmalloc((size_t)n * sizeof(Real));
    Real *expected = xmalloc((size_t)n * sizeof(Real));
    Real *obtained = xmalloc((size_t)n * sizeof(Real));
    Real *scratch = xmalloc((size_t)n * sizeof(Real));

    build_system(n, a, b, c, f);
    thomas_solve(n, a, b, c, f, expected, scratch);
    schur_solve(n, blocks, a, b, c, f, obtained);

    Real worst = largest_difference(n, expected, obtained);
    int failed = !(worst <= TOLERANCE);

    int first_begin;
    int first_end;
    int last_begin;
    int last_end;
    decomp_share(n, blocks, 0, &first_begin, &first_end);
    decomp_share(n, blocks, blocks - 1, &last_begin, &last_end);

    printf("  n = %-4d blocks = %-2d  sizes %d..%d   max |difference| = %.2e  %s\n",
           n, blocks, first_end - first_begin, last_end - last_begin,
           (double)worst, failed ? "FAILED" : "ok");

    free(scratch);
    free(obtained);
    free(expected);
    free(f);
    free(c);
    free(b);
    free(a);

    return failed;
}

int main(void)
{
    par_init(NULL, NULL);

    /* 100 and 97 are not multiples of the block counts, so the blocks come
     * out uneven and the last one is shorter: that is where an off-by-one in
     * the interface bookkeeping shows up. */
    const int sizes[] = {64, 100, 97, 16};
    const int block_counts[] = {1, 2, 3, 4, 5, 8};
    int failed = 0;

    printf("\nSchur complement against sequential Thomas:\n");

    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (size_t t = 0; t < sizeof(block_counts) / sizeof(block_counts[0]);
             t++) {
            int n = sizes[s];
            int blocks = block_counts[t];

            /* Every block needs one internal point besides its interface. */
            if (n / blocks < 2) {
                continue;
            }

            failed |= check(n, blocks);
        }
    }

    printf("\n  %s\n", failed ? "FAILED: the block solve disagrees"
                              : "PASSED: same answer as sequential Thomas");

    par_finalize();
    return failed ? 1 : 0;
}
