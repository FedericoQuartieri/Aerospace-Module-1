/*
 * Do the process blocks cover the grid exactly once?
 *
 * Every process now believes it owns a piece of the grid.  Before any of them
 * computes anything, the pieces must fit together: no cell left out, no cell
 * claimed twice, and the walls recognised only by the processes that actually
 * touch them.
 *
 * The check is one number.  Each process adds up the global linear index of
 * the cells it owns; summed over the processes that has to equal the sum of
 * every index in the grid, which is known in closed form.  A gap lowers it, an
 * overlap raises it, and a block placed at the wrong offset moves it either
 * way, so a single comparison catches all three.
 */
#include <stdio.h>
#include <stdlib.h>

#include "decomp.h"
#include "parallel.h"
#include "solver.h"

/* Sum of the global linear index of every cell this process owns. */
static long long owned_index_sum(const Decomp *d) {
    long long plane = (long long)d->n_global[0] * d->n_global[1];
    long long total = 0;

    for (int k = 0; k < d->n[2]; k++) {
        long long gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            long long gj = decomp_global(d, j, 1);

            for (int i = 0; i < d->n[0]; i++) {
                long long gi = decomp_global(d, i, 0);
                total += gk * plane + gj * d->n_global[0] + gi;
            }
        }
    }

    return total;
}

static int report(const char *what, long long got, long long want) {
    int failed = (got != want);

    if (par_rank() == 0) {
        printf("  %-22s got %lld, expected %lld   %s\n",
               what, got, want, failed ? "FAILED" : "ok");
    }

    return failed;
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    /* Default {1, 1, 0}: MPI only chooses along Z, so the blocks are slabs.
     * Pass three numbers to ask for another shape, 0 meaning "you choose". */
    int process_grid[3] = {1, 1, 0};
    if (argc == 4) {
        for (int c = 0; c < 3; c++) {
            process_grid[c] = atoi(argv[c + 1]);
        }
    }
    par_topology_init(process_grid);

    Decomp d;
    decomp_init_mpi(&d);

    int dims[3];
    int coords[3];
    par_dims(dims);
    par_coords(coords);

    long long cells = (long long)d.n[0] * d.n[1] * d.n[2];
    long long total_cells = (long long)d.n_global[0] * d.n_global[1] *
                            d.n_global[2];

    if (par_rank() == 0) {
        printf("\nProcess grid %d x %d x %d over a %d x %d x %d grid\n",
               dims[0], dims[1], dims[2],
               d.n_global[0], d.n_global[1], d.n_global[2]);
    }

    /* One line per process; the order they appear in is up to MPI. */
    printf("  rank %2d at (%d,%d,%d)  owns %d x %d x %d from (%d,%d,%d)"
           "  walls %d%d%d/%d%d%d  neighbours z: %d %d\n",
           par_rank(), coords[0], coords[1], coords[2],
           d.n[0], d.n[1], d.n[2], d.start[0], d.start[1], d.start[2],
           d.is_first[0], d.is_first[1], d.is_first[2],
           d.is_last[0], d.is_last[1], d.is_last[2],
           par_neighbor(2, -1), par_neighbor(2, +1));
    fflush(stdout);

    int failed = 0;
    failed |= report("cells owned", par_sum_long(cells), total_cells);
    failed |= report("index sum",
                     par_sum_long(owned_index_sum(&d)),
                     total_cells * (total_cells - 1) / 2);

    /*
     * A process must claim a wall exactly when its block ends on one.  This
     * ties each flag to the block it describes; merely counting how many
     * processes raise each flag would not, because the two ends of a
     * direction always hold the same number of processes.
     */
    long long wrong_flags = 0;
    for (int c = 0; c < 3; c++) {
        wrong_flags += (d.is_first[c] != (d.start[c] == 0));
        wrong_flags += (d.is_last[c] !=
                        (d.start[c] + d.n[c] == d.n_global[c]));
    }
    failed |= report("wall flags disagreeing", par_sum_long(wrong_flags), 0);

    /* And a neighbour must be there exactly when the wall is not. */
    long long wrong_neighbours = 0;
    for (int c = 0; c < 3; c++) {
        wrong_neighbours +=
            (d.is_first[c] != (par_neighbor(c, -1) == PAR_NO_NEIGHBOR));
        wrong_neighbours +=
            (d.is_last[c] != (par_neighbor(c, +1) == PAR_NO_NEIGHBOR));
    }
    failed |= report("neighbours disagreeing",
                     par_sum_long(wrong_neighbours), 0);

    if (par_rank() == 0) {
        printf("\n  %s\n", failed ? "FAILED: the blocks do not tile the grid"
                                  : "PASSED: the blocks tile the grid exactly");
    }

    par_finalize();
    return failed ? 1 : 0;
}
