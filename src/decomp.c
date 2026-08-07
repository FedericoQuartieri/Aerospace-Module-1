#include "decomp.h"
#include "parallel.h"
#include "solver.h"

void decomp_share(int total, int parts, int index, int *begin, int *end) {
    int share = total / parts;
    int leftover = total % parts;

    *begin = index * share + (index < leftover ? index : leftover);
    *end = *begin + share + (index < leftover ? 1 : 0);
}

/*
 * Choose where the owned cells sit in memory.  A margin of `halo` cells is
 * left on every face, so consecutive rows and planes are further apart and
 * the first owned cell no longer starts at offset zero.
 *
 * This is the only place that has to change when ghost cells start carrying
 * real data: the kernels reach a cell through stride[] and base alone.
 */
static void decomp_set_layout(Decomp *d, int halo) {
    size_t allocated_x = (size_t)d->n[0] + 2 * (size_t)halo;
    size_t allocated_y = (size_t)d->n[1] + 2 * (size_t)halo;
    size_t allocated_z = (size_t)d->n[2] + 2 * (size_t)halo;

    d->stride[0] = 1;
    d->stride[1] = allocated_x;
    d->stride[2] = allocated_x * allocated_y;
    d->base = (size_t)halo * (d->stride[0] + d->stride[1] + d->stride[2]);
    d->n_cells = allocated_x * allocated_y * allocated_z;
    d->halo = halo;
}

void decomp_init_serial(Decomp *d) {
    decomp_init_serial_padded(d, 0);
}

void decomp_init_serial_padded(Decomp *d, int halo) {
    d->n_global[0] = WIDTH;
    d->n_global[1] = HEIGHT;
    d->n_global[2] = DEPTH;

    for (int c = 0; c < 3; c++) {
        d->n[c] = d->n_global[c];
        d->start[c] = 0;
        d->is_first[c] = 1;
        d->is_last[c] = 1;
    }

    decomp_set_layout(d, halo);
}

void decomp_init_mpi(Decomp *d) {
    int dims[3];
    int coords[3];

    par_dims(dims);
    par_coords(coords);

    d->n_global[0] = WIDTH;
    d->n_global[1] = HEIGHT;
    d->n_global[2] = DEPTH;

    for (int c = 0; c < 3; c++) {
        if (dims[c] > d->n_global[c]) {
            fprintf(stderr,
                    "decomp_init_mpi: %d processes along direction %d but "
                    "only %d cells\n", dims[c], c, d->n_global[c]);
            exit(1);
        }

        int begin;
        int end;
        decomp_share(d->n_global[c], dims[c], coords[c], &begin, &end);

        d->n[c] = end - begin;
        d->start[c] = begin;
        /* Only the blocks at the two ends of a direction touch the walls. */
        d->is_first[c] = (coords[c] == 0);
        d->is_last[c] = (coords[c] == dims[c] - 1);
    }

    /*
     * One ring of ghost cells on every face: the stencils reach one cell past
     * the block, and par_exchange_halo keeps that ring up to date.  A process
     * with no neighbour on a face simply never fills nor reads its ring there.
     */
    decomp_set_layout(d, 1);
}
