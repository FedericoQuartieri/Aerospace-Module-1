#include "schur.h"
#include "parallel.h"
#include "utils.h"

#include <string.h>

void thomas_solve(int n,
                  const Real *a, const Real *b, const Real *c,
                  const Real *f, Real *x, Real *scratch) {
    /*
     * Forward elimination: scratch keeps the normalized superdiagonal.  The
     * reciprocal is taken once and multiplied twice, which is one division
     * instead of two and is what the solver kernels already did inline, so
     * routing them through here does not change a single bit.
     */
    Real inverse = (Real)1 / b[0];

    scratch[0] = c[0] * inverse;
    x[0] = f[0] * inverse;

    for (int i = 1; i < n; i++) {
        inverse = (Real)1 / (b[i] - a[i] * scratch[i - 1]);
        scratch[i] = c[i] * inverse;
        x[i] = (f[i] - a[i] * x[i - 1]) * inverse;
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
 * of a shared array, and `lines` independent systems solved together.
 *
 * The three phases are the same as above, and only the middle one talks to
 * anybody:
 *
 *   1. three local Thomas solves per line          no communication
 *   2. build and solve the interface systems       two messages, for ALL lines
 *   3. recombine the local answers                 no communication
 *
 * Batching matters: the solver calls this once per grid line, and one message
 * per line would cost far more than the arithmetic it enables.  Doing the
 * local work for a whole group of lines first, and then exchanging once for
 * the whole group, turns thousands of tiny messages into one.
 *
 * System `l` occupies a[l*n_local] .. a[l*n_local + n_local - 1], and likewise
 * for b, c, f and x.
 *
 * Solving the interface system on every process duplicates a little work, but
 * it is a system of `blocks - 1` unknowns against blocks of hundreds of
 * points, and it saves a second collective.
 */
void schur_solve_mpi(int axis, int lines, int n_local,
                     const Real *a, const Real *b, const Real *c,
                     const Real *f, Real *x) {
    int dims[3];
    int coords[3];

    par_dims(dims);
    par_coords(coords);

    int blocks = dims[axis];
    int p = coords[axis];

    size_t line_bytes = (size_t)n_local * sizeof(Real);
    Real *scratch = xmalloc(line_bytes);

    if (blocks < 2) {
        for (int l = 0; l < lines; l++) {
            size_t off = (size_t)l * (size_t)n_local;
            thomas_solve(n_local, a + off, b + off, c + off,
                         f + off, x + off, scratch);
        }
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

    size_t total = (size_t)lines * (size_t)n_local;
    Real *y = xmalloc(total * sizeof(Real));
    Real *lft = xmalloc(total * sizeof(Real));
    Real *rgt = xmalloc(total * sizeof(Real));
    Real *rhs = xmalloc(line_bytes);

    memset(lft, 0, total * sizeof(Real));
    memset(rgt, 0, total * sizeof(Real));

    /* ---- 1. every line on its own ---- */
    for (int l = 0; l < lines; l++) {
        size_t off = (size_t)l * (size_t)n_local;

        thomas_solve(len, a + off, b + off, c + off,
                     f + off, y + off, scratch);

        if (p > 0) {
            memset(rhs, 0, (size_t)len * sizeof(Real));
            rhs[0] = a[off];
            thomas_solve(len, a + off, b + off, c + off,
                         rhs, lft + off, scratch);
        }

        if (has_right) {
            memset(rhs, 0, (size_t)len * sizeof(Real));
            rhs[len - 1] = c[off + (size_t)len - 1];
            thomas_solve(len, a + off, b + off, c + off,
                         rhs, rgt + off, scratch);
        }
    }

    /* ---- 2. one exchange and one collective, for every line at once ---- */
    Real *mine = xmalloc(3 * (size_t)lines * sizeof(Real));
    Real *from_right = xmalloc(3 * (size_t)lines * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        size_t off = (size_t)l * (size_t)n_local;

        mine[3 * l + 0] = y[off];
        mine[3 * l + 1] = lft[off];
        mine[3 * l + 2] = rgt[off];
        from_right[3 * l + 0] = (Real)0;
        from_right[3 * l + 1] = (Real)0;
        from_right[3 * l + 2] = (Real)0;
    }

    par_shift_real(axis, -1, mine, from_right, 3 * lines);

    /* My row of every interface system.  The last process has no interface
     * and sends harmless identity rows that nobody uses. */
    Real *row = xmalloc(4 * (size_t)lines * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        size_t off = (size_t)l * (size_t)n_local;

        row[4 * l + 0] = (Real)0;
        row[4 * l + 1] = (Real)1;
        row[4 * l + 2] = (Real)0;
        row[4 * l + 3] = (Real)0;

        if (has_right) {
            size_t m = off + (size_t)n_local - 1;  /* the interface point   */
            size_t left = off + (size_t)len - 1;   /* my last internal point */

            row[4 * l + 0] = -a[m] * lft[left];
            row[4 * l + 1] = b[m] - a[m] * rgt[left]
                             - c[m] * from_right[3 * l + 1];
            row[4 * l + 2] = -c[m] * from_right[3 * l + 2];
            row[4 * l + 3] = f[m] - a[m] * y[left]
                             - c[m] * from_right[3 * l + 0];
        }
    }

    Real *rows = xmalloc(4 * (size_t)lines * (size_t)blocks * sizeof(Real));
    par_line_allgather(axis, row, 4 * lines, rows);

    /* ---- 3. every line on its own again ---- */
    int interfaces = blocks - 1;
    Real *ra = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rb = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rc = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rf = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rx = xmalloc((size_t)interfaces * sizeof(Real));
    Real *reduced_scratch = xmalloc((size_t)interfaces * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        for (int q = 0; q < interfaces; q++) {
            size_t here = 4 * (size_t)q * (size_t)lines + 4 * (size_t)l;

            ra[q] = rows[here + 0];
            rb[q] = rows[here + 1];
            rc[q] = rows[here + 2];
            rf[q] = rows[here + 3];
        }

        thomas_solve(interfaces, ra, rb, rc, rf, rx, reduced_scratch);

        Real x_left = (p > 0) ? rx[p - 1] : (Real)0;
        Real x_right = has_right ? rx[p] : (Real)0;
        size_t off = (size_t)l * (size_t)n_local;

        for (int i = 0; i < len; i++) {
            x[off + (size_t)i] = y[off + (size_t)i]
                                 - x_left * lft[off + (size_t)i]
                                 - x_right * rgt[off + (size_t)i];
        }

        if (has_right) {
            x[off + (size_t)n_local - 1] = rx[p];
        }
    }

    free(reduced_scratch);
    free(rx);
    free(rf);
    free(rc);
    free(rb);
    free(ra);
    free(rows);
    free(row);
    free(from_right);
    free(mine);
    free(rhs);
    free(rgt);
    free(lft);
    free(y);
    free(scratch);
}

/*
 * Preprocessing (Lecture 5, p. 32, step 1).
 *
 * Everything here depends on the matrix alone: the two influence functions, the
 * values the block above contributes to my interface equation, and the
 * resulting interface system, which every process assembles whole.
 */
void schur_plan_init(SchurPlan *plan, int axis, int n,
                     const Real *a, const Real *b, const Real *c) {
    int dims[3];
    int coords[3];

    par_dims(dims);
    par_coords(coords);

    size_t bytes = (size_t)n * sizeof(Real);

    plan->axis = axis;
    plan->n = n;
    plan->blocks = dims[axis];
    plan->p = coords[axis];
    plan->has_right = (plan->p < plan->blocks - 1);
    plan->len = plan->has_right ? n - 1 : n;

    plan->a = xmalloc(bytes);
    plan->b = xmalloc(bytes);
    plan->c = xmalloc(bytes);
    memcpy(plan->a, a, bytes);
    memcpy(plan->b, b, bytes);
    memcpy(plan->c, c, bytes);

    plan->lft = NULL;
    plan->rgt = NULL;
    plan->ra = NULL;
    plan->rb = NULL;
    plan->rc = NULL;

    /* One process along the axis: the runtime is plain Thomas on the whole
     * line, so there is nothing to prepare. */
    if (plan->blocks < 2) {
        return;
    }

    if (plan->len < 1) {
        fprintf(stderr,
                "schur_plan_init: block %d of %d holds only %d points\n",
                plan->p, plan->blocks, n);
        exit(1);
    }

    int len = plan->len;

    plan->lft = xmalloc(bytes);
    plan->rgt = xmalloc(bytes);
    memset(plan->lft, 0, bytes);
    memset(plan->rgt, 0, bytes);

    Real *rhs = xmalloc(bytes);
    Real *scratch = xmalloc(bytes);

    if (plan->p > 0) {
        memset(rhs, 0, (size_t)len * sizeof(Real));
        rhs[0] = a[0];
        thomas_solve(len, a, b, c, rhs, plan->lft, scratch);
    }

    if (plan->has_right) {
        memset(rhs, 0, (size_t)len * sizeof(Real));
        rhs[len - 1] = c[len - 1];
        thomas_solve(len, a, b, c, rhs, plan->rgt, scratch);
    }

    /* My interface equation also reaches the first internal point of the block
     * above, through that block's own influence functions. */
    Real mine[2] = { plan->lft[0], plan->rgt[0] };
    Real above[2] = { (Real)0, (Real)0 };

    par_shift_real(axis, -1, mine, above, 2);

    /* My row of the interface system.  The last process owns no interface and
     * contributes a harmless identity row that nobody uses. */
    Real row[3] = { (Real)0, (Real)1, (Real)0 };

    if (plan->has_right) {
        int m = n - 1;        /* the interface point itself */
        int left = len - 1;   /* my last internal point     */

        row[0] = -a[m] * plan->lft[left];
        row[1] = b[m] - a[m] * plan->rgt[left] - c[m] * above[0];
        row[2] = -c[m] * above[1];
    }

    int interfaces = plan->blocks - 1;
    Real *rows = xmalloc(3 * (size_t)plan->blocks * sizeof(Real));

    par_line_allgather(axis, row, 3, rows);

    plan->ra = xmalloc((size_t)interfaces * sizeof(Real));
    plan->rb = xmalloc((size_t)interfaces * sizeof(Real));
    plan->rc = xmalloc((size_t)interfaces * sizeof(Real));

    for (int q = 0; q < interfaces; q++) {
        plan->ra[q] = rows[3 * (size_t)q + 0];
        plan->rb[q] = rows[3 * (size_t)q + 1];
        plan->rc[q] = rows[3 * (size_t)q + 2];
    }

    free(rows);
    free(scratch);
    free(rhs);
}

/*
 * Runtime (Lecture 5, p. 32, step 2).  Only the right-hand side is new: one
 * local solve per line, one value per line exchanged and gathered, then the
 * interface answers are folded back into the local ones.
 */
void schur_plan_solve(const SchurPlan *plan, int lines,
                      const Real *f, Real *x) {
    const Real *a = plan->a;
    const Real *b = plan->b;
    const Real *c = plan->c;
    const int n = plan->n;
    const int len = plan->len;

    Real *scratch = xmalloc((size_t)n * sizeof(Real));

    if (plan->blocks < 2) {
        for (int l = 0; l < lines; l++) {
            size_t off = (size_t)l * (size_t)n;
            thomas_solve(n, a, b, c, f + off, x + off, scratch);
        }
        free(scratch);
        return;
    }

    Real *y = xmalloc((size_t)lines * (size_t)n * sizeof(Real));

    /* ---- 1. every line against its own right-hand side ---- */
    for (int l = 0; l < lines; l++) {
        size_t off = (size_t)l * (size_t)n;
        thomas_solve(len, a, b, c, f + off, y + off, scratch);
    }

    /* ---- 2. one exchange and one collective, for every line at once ---- */
    Real *mine = xmalloc((size_t)lines * sizeof(Real));
    Real *above = xmalloc((size_t)lines * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        mine[l] = y[(size_t)l * (size_t)n];
        above[l] = (Real)0;
    }

    par_shift_real(plan->axis, -1, mine, above, lines);

    Real *rf = xmalloc((size_t)lines * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        rf[l] = (Real)0;

        if (plan->has_right) {
            size_t off = (size_t)l * (size_t)n;

            rf[l] = f[off + (size_t)n - 1]
                    - a[n - 1] * y[off + (size_t)len - 1]
                    - c[n - 1] * above[l];
        }
    }

    Real *rows = xmalloc((size_t)lines * (size_t)plan->blocks * sizeof(Real));
    par_line_allgather(plan->axis, rf, lines, rows);

    /* ---- 3. every line on its own again ---- */
    int interfaces = plan->blocks - 1;
    Real *rline = xmalloc((size_t)interfaces * sizeof(Real));
    Real *rx = xmalloc((size_t)interfaces * sizeof(Real));
    Real *reduced_scratch = xmalloc((size_t)interfaces * sizeof(Real));

    for (int l = 0; l < lines; l++) {
        for (int q = 0; q < interfaces; q++) {
            rline[q] = rows[(size_t)q * (size_t)lines + (size_t)l];
        }

        thomas_solve(interfaces, plan->ra, plan->rb, plan->rc,
                     rline, rx, reduced_scratch);

        Real x_left = (plan->p > 0) ? rx[plan->p - 1] : (Real)0;
        Real x_right = plan->has_right ? rx[plan->p] : (Real)0;
        size_t off = (size_t)l * (size_t)n;

        for (int i = 0; i < len; i++) {
            x[off + (size_t)i] = y[off + (size_t)i]
                                 - x_left * plan->lft[i]
                                 - x_right * plan->rgt[i];
        }

        if (plan->has_right) {
            x[off + (size_t)n - 1] = rx[plan->p];
        }
    }

    free(reduced_scratch);
    free(rx);
    free(rline);
    free(rows);
    free(rf);
    free(above);
    free(mine);
    free(y);
    free(scratch);
}

void schur_plan_free(SchurPlan *plan) {
    free(plan->rc);
    free(plan->rb);
    free(plan->ra);
    free(plan->rgt);
    free(plan->lft);
    free(plan->c);
    free(plan->b);
    free(plan->a);
}
