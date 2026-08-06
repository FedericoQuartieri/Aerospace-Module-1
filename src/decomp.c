#include "decomp.h"
#include "solver.h"

/*
 * Single-block decomposition: the process owns the whole grid, so local and
 * global indices coincide and the strides are the ones the kernels used to
 * build inline from WIDTH and HEIGHT.
 *
 * A distributed decomposition will differ only here: n[] shrinks, start[]
 * becomes the block origin, is_first/is_last stop being uniformly true, and
 * stride[]/base/n_cells grow to account for the ghost cells.
 */
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

    /* Allocated extents include the margin on both faces of each direction. */
    size_t allocated_x = (size_t)d->n[0] + 2 * (size_t)halo;
    size_t allocated_y = (size_t)d->n[1] + 2 * (size_t)halo;
    size_t allocated_z = (size_t)d->n[2] + 2 * (size_t)halo;

    d->stride[0] = 1;
    d->stride[1] = allocated_x;
    d->stride[2] = allocated_x * allocated_y;
    d->base = (size_t)halo * (d->stride[0] + d->stride[1] + d->stride[2]);
    d->n_cells = allocated_x * allocated_y * allocated_z;
}
