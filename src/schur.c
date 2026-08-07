#include "schur.h"
#include "parallel.h"
#include "utils.h"

#include <string.h>

void thomas_solve(int n,
                  const Real *a, const Real *b, const Real *c,
                  const Real *f, Real *x, Real *scratch) {
    /* Forward elimination: scratch keeps the normalized superdiagonal. */
    Real denominator = b[0];

    scratch[0] = c[0] / denominator;
    x[0] = f[0] / denominator;

    for (int i = 1; i < n; i++) {
        denominator = b[i] - a[i] * scratch[i - 1];
        scratch[i] = c[i] / denominator;
        x[i] = (f[i] - a[i] * x[i - 1]) / denominator;
    }

    /* Back substitution. */
    for (int i = n - 2; i >= 0; i--) {
        x[i] -= scratch[i] * x[i + 1];
    }
}

/*
 * The last point of every block but the last is shared with the next block:
 * those are the interface unknowns, blocks - 1 of them.  Everything else is
 * internal to exactly one block.
 *
 *   block 0            block 1            block 2
 *   [ internal ][I]    [ internal ][I]    [ internal ]
 *                ^                  ^
 *                interface 0        interface 1
 *
 * Inside a block the internal equations only reach outside through the two
 * interfaces bounding it, so the internal solution is
 *
 *   x_internal = y  -  x_left * Lft  -  x_right * Rgt
 *
 * with y the answer to the block's own right-hand side and Lft, Rgt the
 * answers to a unit value sitting on the left and right interface.  Three
 * local Thomas solves, no coupling.  Substituting that expression into the
 * equations written at the interfaces leaves a system in the interface
 * unknowns alone, which is again tridiagonal.
 */
void schur_solve(int n, int blocks,
                 const Real *a, const Real *b, const Real *c,
                 const Real *f, Real *x) {
    if (blocks < 2) {
        Real *scratch = xmalloc((size_t)n * sizeof(Real));
        thomas_solve(n, a, b, c, f, x, scratch);
        free(scratch);
        return;
    }

    int interfaces = blocks - 1;
    size_t bytes = (size_t)n * sizeof(Real);

    Real *y = xmalloc(bytes);
    Real *lft = xmalloc(bytes);
    Real *rgt = xmalloc(bytes);
    Real *rhs = xmalloc(bytes);
    Real *scratch = xmalloc(bytes);

    /* A block without a left (right) interface leaves lft (rgt) at zero. */
    memset(lft, 0, bytes);
    memset(rgt, 0, bytes);

    /* ---- 1. every block on its own ---- */
    for (int p = 0; p < blocks; p++) {
        int begin;
        int end;
        decomp_share(n, blocks, p, &begin, &end);

        int has_right = (p < blocks - 1);
        int internal_end = has_right ? end - 1 : end;
        int len = internal_end - begin;

        if (len < 1) {
            fprintf(stderr,
                    "schur_solve: block %d of %d is too small for n = %d\n",
                    p, blocks, n);
            exit(1);
        }

        thomas_solve(len, a + begin, b + begin, c + begin,
                     f + begin, y + begin, scratch);

        if (p > 0) {
            memset(rhs, 0, (size_t)len * sizeof(Real));
            rhs[0] = a[begin];
            thomas_solve(len, a + begin, b + begin, c + begin,
                         rhs, lft + begin, scratch);
        }

        if (has_right) {
            memset(rhs, 0, (size_t)len * sizeof(Real));
            rhs[len - 1] = c[internal_end - 1];
            thomas_solve(len, a + begin, b + begin, c + begin,
                         rhs, rgt + begin, scratch);
        }
    }

    /* ---- 2. the system left on the interfaces ---- */
    Real *ra = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rb = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rc = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rf = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rx = xmalloc((size_t)interfaces * sizeof(Real));

    for (int p = 0; p < interfaces; p++) {
        int begin;
        int end;
        int next_begin;
        int next_end;
        decomp_share(n, blocks, p, &begin, &end);
        decomp_share(n, blocks, p + 1, &next_begin, &next_end);

        int m = end - 1;         /* the interface point itself           */
        int left = m - 1;        /* last internal point of block p       */
        int right = next_begin;  /* first internal point of block p + 1  */

        ra[p] = -a[m] * lft[left];
        rb[p] = b[m] - a[m] * rgt[left] - c[m] * lft[right];
        rc[p] = -c[m] * rgt[right];
        rf[p] = f[m] - a[m] * y[left] - c[m] * y[right];
    }

    thomas_solve(interfaces, ra, rb, rc, rf, rx, scratch);

    /* ---- 3. every block again on its own ---- */
    for (int p = 0; p < blocks; p++) {
        int begin;
        int end;
        decomp_share(n, blocks, p, &begin, &end);

        int has_right = (p < blocks - 1);
        int internal_end = has_right ? end - 1 : end;

        Real x_left = (p > 0) ? rx[p - 1] : (Real)0;
        Real x_right = has_right ? rx[p] : (Real)0;

        for (int i = begin; i < internal_end; i++) {
            x[i] = y[i] - x_left * lft[i] - x_right * rgt[i];
        }

        if (has_right) {
            x[end - 1] = rx[p];
        }
    }

    free(rx);
    free(rf);
    free(rc);
    free(rb);
    free(ra);
    free(scratch);
    free(rhs);
    free(rgt);
    free(lft);
    free(y);
}

/*
 * The distributed form: one block per process instead of one block per slice
 * of a shared array.  The three phases are the same as above, and only the
 * middle one talks to anybody:
 *
 *   1. three local Thomas solves                        no communication
 *   2. build and solve the interface system             two small messages
 *   3. recombine the local answer                       no communication
 *
 * Phase 2 needs, from the process on the right, the three values its first
 * internal point takes in y, Lft and Rgt: that is one exchange of three
 * numbers.  Then each process holds one row of the interface system, and an
 * allgather of four numbers per process gives everybody the whole thing, so
 * everybody solves it and nobody has to broadcast the answer back.
 *
 * Solving the interface system on every process duplicates a little work, but
 * it is a system of `blocks - 1` unknowns against blocks of hundreds of
 * points, and it saves a second collective.  It stops paying off only for
 * very large process counts.
 */
void schur_solve_mpi(int axis, int n_local,
                     const Real *a, const Real *b, const Real *c,
                     const Real *f, Real *x) {
    int dims[3];
    int coords[3];

    par_dims(dims);
    par_coords(coords);

    int blocks = dims[axis];
    int p = coords[axis];

    size_t bytes = (size_t)n_local * sizeof(Real);
    Real *scratch = xmalloc(bytes);

    if (blocks < 2) {
        thomas_solve(n_local, a, b, c, f, x, scratch);
        free(scratch);
        return;
    }

    /* The last point of a block is the interface it shares with the next. */
    int has_right = (p < blocks - 1);
    int len = has_right ? n_local - 1 : n_local;

    if (len < 1) {
        fprintf(stderr,
                "schur_solve_mpi: block %d of %d holds only %d points\n",
                p, blocks, n_local);
        exit(1);
    }

    Real *y = xmalloc(bytes);
    Real *lft = xmalloc(bytes);
    Real *rgt = xmalloc(bytes);
    Real *rhs = xmalloc(bytes);

    memset(lft, 0, bytes);
    memset(rgt, 0, bytes);

    /* ---- 1. on my own ---- */
    thomas_solve(len, a, b, c, f, y, scratch);

    if (p > 0) {
        memset(rhs, 0, (size_t)len * sizeof(Real));
        rhs[0] = a[0];
        thomas_solve(len, a, b, c, rhs, lft, scratch);
    }

    if (has_right) {
        memset(rhs, 0, (size_t)len * sizeof(Real));
        rhs[len - 1] = c[len - 1];
        thomas_solve(len, a, b, c, rhs, rgt, scratch);
    }

    /* ---- 2. the interface system ---- */

    /* My left neighbour needs what my first internal point is worth. */
    Real mine[3] = {y[0], lft[0], rgt[0]};
    Real from_right[3] = {0, 0, 0};
    par_shift_real(axis, -1, mine, from_right, 3);

    /* Row of the interface system owned by this process.  The last process
     * has no interface, and sends a harmless identity row that nobody uses. */
    Real row[4] = {0, 1, 0, 0};

    if (has_right) {
        int m = n_local - 1;  /* the interface point   */
        int left = len - 1;   /* my last internal point */

        row[0] = -a[m] * lft[left];
        row[1] = b[m] - a[m] * rgt[left] - c[m] * from_right[1];
        row[2] = -c[m] * from_right[2];
        row[3] = f[m] - a[m] * y[left] - c[m] * from_right[0];
    }

    Real *rows = xmalloc(4 * (size_t)blocks * sizeof(Real));
    par_line_allgather(axis, row, 4, rows);

    int interfaces = blocks - 1;
    Real *ra = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rb = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rc = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rf = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rx = xmalloc((size_t)interfaces * sizeof(Real));
    Real *reduced_scratch = xmalloc((size_t)interfaces * sizeof(Real));

    for (int q = 0; q < interfaces; q++) {
        ra[q] = rows[4 * q + 0];
        rb[q] = rows[4 * q + 1];
        rc[q] = rows[4 * q + 2];
        rf[q] = rows[4 * q + 3];
    }

    thomas_solve(interfaces, ra, rb, rc, rf, rx, reduced_scratch);

    /* ---- 3. on my own again ---- */
    Real x_left = (p > 0) ? rx[p - 1] : (Real)0;
    Real x_right = has_right ? rx[p] : (Real)0;

    for (int i = 0; i < len; i++) {
        x[i] = y[i] - x_left * lft[i] - x_right * rgt[i];
    }

    if (has_right) {
        x[n_local - 1] = rx[p];
    }

    free(reduced_scratch);
    free(rx);
    free(rf);
    free(rc);
    free(rb);
    free(ra);
    free(rows);
    free(rhs);
    free(rgt);
    free(lft);
    free(y);
    free(scratch);
}
