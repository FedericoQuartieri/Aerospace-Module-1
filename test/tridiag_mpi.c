/*
 * The same check as tridiag_blocks, with the blocks on different processes.
 *
 * Every process builds the whole system from the same seed and solves it
 * sequentially, which costs nothing at these sizes and gives an exact
 * reference.  Then it solves only its own slice through the distributed
 * Schur complement and compares the two.  The answer must agree to round-off
 * whatever the number of processes: the method is direct, so the split
 * changes only the order the sums are taken in.
 *
 * The algebra was already checked in tridiag_blocks and the block layout in
 * decomp_mpi, so anything failing here is the communication.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "decomp.h"
#include "parallel.h"
#include "schur.h"
#include "utils.h"

static unsigned int random_state = 12345u;

static Real next_random(void) {
    random_state = random_state * 1103515245u + 12345u;
    return (Real)((random_state >> 16) & 0x7fffu) / (Real)0x7fffu;
}

/* Same system as tridiag_blocks, so the two tests can be compared. */
static void build_system(int n, Real *a, Real *b, Real *c, Real *f) {
    random_state = 12345u;

    for (int i = 0; i < n; i++) {
        a[i] = -(Real)1 - next_random();
        c[i] = -(Real)1 - next_random();
        b[i] = (Real)4 + next_random();
        f[i] = next_random() - (Real)0.5;
    }
}

#define TOLERANCE ((Real)1e-12)
#define AXIS 2

static int check(int n, int blocks, int block) {
    Real *a = xmalloc((size_t)n * sizeof(Real));
    Real *b = xmalloc((size_t)n * sizeof(Real));
    Real *c = xmalloc((size_t)n * sizeof(Real));
    Real *f = xmalloc((size_t)n * sizeof(Real));
    Real *expected = xmalloc((size_t)n * sizeof(Real));
    Real *scratch = xmalloc((size_t)n * sizeof(Real));

    build_system(n, a, b, c, f);
    thomas_solve(n, a, b, c, f, expected, scratch);

    int begin;
    int end;
    decomp_share(n, blocks, block, &begin, &end);
    int n_local = end - begin;

    Real *mine = xmalloc((size_t)n_local * sizeof(Real));
    schur_solve_mpi(AXIS, 1, n_local, a + begin, b + begin, c + begin,
                    f + begin, mine);

    Real worst = (Real)0;
    for (int i = 0; i < n_local; i++) {
        Real difference =
            (Real)fabs((double)(mine[i] - expected[begin + i]));
        /* Written inverted so that a NaN, whose comparisons are all
         * false, becomes the worst value instead of being skipped. */
        if (!(difference <= worst)) {
            worst = difference;
        }
    }

    /*
     * Judge locally first.  Writing it as !(worst <= TOLERANCE) makes a NaN
     * count as a failure, and summing the verdicts keeps it that way: MPI_MAX
     * would quietly drop the NaN, because every comparison against it is
     * false, and the run would look clean.
     */
    int failed = !(worst <= TOLERANCE);
    failed = (par_sum_long(failed) != 0);

    /* Only for the report, so a failure says how far off it went. */
    worst = par_max_real(worst);

    if (par_rank() == 0) {
        printf("  n = %-4d processes = %-2d   max |difference| = %.2e  %s\n",
               n, blocks, (double)worst, failed ? "FAILED" : "ok");
    }

    free(mine);
    free(scratch);
    free(expected);
    free(f);
    free(c);
    free(b);
    free(a);

    return failed;
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    const int process_grid[3] = {1, 1, 0};
    par_topology_init(process_grid);

    int dims[3];
    int coords[3];
    par_dims(dims);
    par_coords(coords);

    const int sizes[] = {64, 100, 97, 16};
    int failed = 0;

    if (par_rank() == 0) {
        printf("\nDistributed Schur complement against sequential Thomas:\n");
    }

    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];

        /* Every block needs one internal point besides its interface. */
        if (n / dims[AXIS] < 2) {
            if (par_rank() == 0) {
                printf("  n = %-4d processes = %-2d   skipped, blocks "
                       "would be too short\n", n, dims[AXIS]);
            }
            continue;
        }

        failed |= check(n, dims[AXIS], coords[AXIS]);
    }

    if (par_rank() == 0) {
        printf("\n  %s\n", failed ? "FAILED: the distributed solve disagrees"
                                  : "PASSED: same answer as sequential Thomas");
    }

    par_finalize();
    return failed ? 1 : 0;
}
