/*
 * Do the boundary cells really contain the neighbour's data?
 *
 * Each process fills the cells it owns with a number that depends only on the
 * position of the cell in the domain, and puts a recognisable value in the
 * boundary ring. After the exchange, every cell of the ring must contain the
 * number that belongs to its position: it is a copy, so the expected error is
 * exactly zero.
 *
 * The opposite case is also checked: where the neighbour does not exist,
 * because the domain ends there, the ring must have stayed intact. If someone
 * wrote data there, the solver would use invented values in place of the
 * boundary conditions.
 */
#include <stdio.h>
#include <stdlib.h>

#include "decomp.h"
#include "parallel.h"
#include "solver.h"
#include "utils.h"

#define SEGNAPOSTO ((Real)-987654.0)

/*
 * Number that identifies a cell by its position in the domain: it is its
 * global index, so two different cells always have different numbers.
 */
static Real tag_of(const Decomp *d, int gi, int gj, int gk) {
    long long plane = (long long)d->n_global[0] * d->n_global[1];
    return (Real)((long long)gk * plane +
                  (long long)gj * d->n_global[0] + gi);
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    int process_grid[3] = {1, 1, 0};
    if (argc == 4) {
        for (int c = 0; c < 3; c++) {
            process_grid[c] = atoi(argv[c + 1]);
        }
    }
    par_topology_init(process_grid);

    Decomp d;
    decomp_init_mpi(&d);

    Real *field = xmalloc(d.n_cells * sizeof(Real));

    /* All placeholder, owned cells included: this way a ring that is not
     * filled stays recognisable. */
    for (size_t i = 0; i < d.n_cells; i++) {
        field[i] = SEGNAPOSTO;
    }

    for (int k = 0; k < d.n[2]; k++) {
        for (int j = 0; j < d.n[1]; j++) {
            for (int i = 0; i < d.n[0]; i++) {
                field[decomp_index(&d, i, j, k)] =
                    tag_of(&d, decomp_global(&d, i, 0),
                           decomp_global(&d, j, 1),
                           decomp_global(&d, k, 2));
            }
        }
    }

    par_exchange_halo(&d, field);

    long long wrong_values = 0;   /* ring filled wrongly */
    long long wrong_walls = 0;    /* ring written where it is not needed */

    for (int axis = 0; axis < 3; axis++) {
        int first = (axis + 1) % 3;
        int second = (axis + 2) % 3;

        /* The two rings: below the first face and above the last. */
        for (int side = 0; side < 2; side++) {
            int slot = (side == 0) ? -1 : d.n[axis];
            int step = (side == 0) ? -1 : +1;
            int neighbor = par_neighbor(axis, step);
            int cell[3];

            cell[axis] = slot;
            for (int b = 0; b < d.n[second]; b++) {
                cell[second] = b;
                for (int a = 0; a < d.n[first]; a++) {
                    cell[first] = a;

                    Real got = field[decomp_index(&d, cell[0], cell[1],
                                                  cell[2])];

                    if (neighbor == PAR_NO_NEIGHBOR) {
                        wrong_walls += (got != SEGNAPOSTO);
                    } else {
                        Real want = tag_of(&d,
                                           decomp_global(&d, cell[0], 0),
                                           decomp_global(&d, cell[1], 1),
                                           decomp_global(&d, cell[2], 2));
                        wrong_values += (got != want);
                    }
                }
            }
        }
    }

    long long bad_values = par_sum_long(wrong_values);
    long long bad_walls = par_sum_long(wrong_walls);
    int failed = (bad_values != 0) || (bad_walls != 0);

    int dims[3];
    par_dims(dims);

    if (par_rank() == 0) {
        printf("\nHalo exchange:\n");
        printf("  processes %d x %d x %d over a %d x %d x %d grid\n",
               dims[0], dims[1], dims[2],
               d.n_global[0], d.n_global[1], d.n_global[2]);
        printf("  wrong halo cells:             %lld\n", bad_values);
        printf("  cells written against a wall: %lld\n", bad_walls);
        printf("\n  %s\n", failed
               ? "FAILED: the halo does not hold the neighbour's data"
               : "PASSED: the halo holds the neighbour's data");
    }

    free(field);
    par_finalize();
    return failed ? 1 : 0;
}
