#ifndef DECOMP_H
#define DECOMP_H

#include <stddef.h>

/*
 * Domain decomposition descriptor.
 *
 * Every routine that walks the grid receives a Decomp instead of reading the
 * WIDTH/HEIGHT/DEPTH macros directly.  The macros still describe the *global*
 * problem (and therefore keep driving DX, DY, DZ and the physical extents),
 * while the fields below describe the block owned by the current process.
 *
 * With a single process the block is the whole grid, so the two coincide and
 * the solver produces exactly the same numbers as before.  Splitting the grid
 * later only changes how this struct is filled, not the code that reads it.
 *
 * Index conventions
 * -----------------
 *   local index   i in [0, n[c])      position inside the owned block
 *   global index  start[c] + i        position in the whole grid; this is what
 *                                     physical coordinates and boundary tests
 *                                     must be derived from
 *
 * Memory layout
 * -------------
 * stride[c] is the distance, in array elements, between two cells that are
 * adjacent along direction c.  Keeping it as data rather than as an expression
 * built from WIDTH and HEIGHT is what will later allow ghost cells to be added
 * by changing decomp_init alone: the indexing code below stays untouched.
 *
 * stride[0] is always 1: X is the contiguous direction, and the numerical
 * kernels rely on it for unit-stride inner loops.
 */
typedef struct Decomp {
    int n_global[3];  /* size of the whole grid                    */
    int n[3];         /* cells owned along each direction          */
    int start[3];     /* global index of the first owned cell      */
    size_t stride[3]; /* element distance between adjacent cells   */
    size_t base;      /* offset of the first owned cell            */
    int is_first[3];  /* block touches the lower global boundary   */
    int is_last[3];   /* block touches the upper global boundary   */
    size_t n_cells;   /* elements to allocate for one field        */
} Decomp;

/* Fill d with the single-block decomposition covering the whole grid. */
void decomp_init_serial(Decomp *d);

/*
 * Same block, but allocated with a margin of halo cells on every face, so the
 * owned cells are no longer contiguous and no longer start at offset zero.
 * The margin stays untouched: this exists to prove that the kernels address
 * memory only through stride[] and base, before ghost cells carry real data.
 */
void decomp_init_serial_padded(Decomp *d, int halo);

/* Offset of the owned cell (i, j, k), all indices local. */
static inline size_t decomp_index(const Decomp *d, int i, int j, int k) {
    return d->base + (size_t)i * d->stride[0] + (size_t)j * d->stride[1] +
           (size_t)k * d->stride[2];
}

/* Global index along direction component of the local index i. */
static inline int decomp_global(const Decomp *d, int i, int component) {
    return d->start[component] + i;
}

#endif
