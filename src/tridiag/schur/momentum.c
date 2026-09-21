#include "momentum.h"
#include "backend.h"
#include "backend_schur.h"
#include "momentum_row.h"
#include "schur.h"
#include "utils.h"
#include "workers.h"

/*
 * The three momentum steps are the same operation repeated on different axes
 * (Lecture 5, p. 6):
 *
 *   (I - g d_xx)(eta^{n+1}  - eta^n)  = xi^{n+1}  - eta^n
 *   (I - g d_yy)(zeta^{n+1} - zeta^n) = eta^{n+1} - zeta^n
 *   (I - g d_zz)(u^{n+1}    - u^n)    = zeta^{n+1} - u^n
 *
 * The axis, the starting field and the arrival field change; the rest is
 * identical. This function does all three, and it is called with the axis.
 *
 * The system of each line is written out in full in the three diagonals and
 * the right-hand side, and passed to schur_solve_mpi: if along that axis there
 * is a single process it uses plain Thomas, otherwise it stitches together the
 * pieces of the split lines. The lines of the same group are solved together,
 * so the group costs one communication instead of one per line.
 */
/*
 * The context of a line: what the loop body must know and does not change from
 * one line to the next.
 *
 * It is needed because the same body is now walked by two different loop
 * structures -- one that distributes the planes among the threads, one that
 * distributes the lines within the plane -- and repeating the same arguments
 * in two places would be the easiest way to make them diverge without
 * noticing.
 *
 * The physics is not in here: it is in `mline`, that is in momentum_row.h,
 * which is shared with the other backend. This structure carries only the
 * geometry.
 */
typedef struct {
    const MomentumLine *mline;
    const Decomp *d;
    Real *target;
    /*
     * The abscissae of the cells of a line along x, or NULL outside the eta
     * step. They are the same for every line of the block, for every component
     * and for every step: they are filled once and the threads just read them.
     */
    const Real *abscissa;
    /*
     * Where each thread adds up the nanoseconds it spends preparing g.
     *
     * One counter per thread and not just one: two threads incrementing the
     * same variable corrupt it, and a single one protected by an atomic would
     * cost more than the thing it measures. They are spaced one cache line
     * apart because neighbouring counters bounce between the cores, and what
     * one would end up measuring is the bouncing.
     */
    uint64_t *g_ns;
    int axis;
    int group;
    int outer;
    int length;
    size_t step;
} MomentumLines;

/* One counter per thread, spaced one cache line apart. */
#define COUNTER_STRIDE 8

/*
 * Writes into the three diagonals and the right-hand side the system of ONE
 * line.
 *
 * `source_term` is the work space where to put the physical term g of the
 * line, or NULL outside the eta step: this function fills it, a single call
 * instead of one per cell, and momentum_row then picks from it.
 */
static void momentum_assemble_line(const MomentumLines *ml, int b, int a,
                                   Real *restrict lower,
                                   Real *restrict diagonal,
                                   Real *restrict upper,
                                   Real *restrict known,
                                   Real *restrict source_term) {
    const Decomp *restrict d = ml->d;
    const int axis = ml->axis;
    const int length = ml->length;
    /*
     * A copy of the shared line, with its own g attached: the structure of
     * momentum_row.h is one per direction, while the piece of scratch belongs
     * to this thread and this line.
     */
    MomentumLine mline = *ml->mline;
    int cell[3];

    // Three nested indices: b (outer) -> a (group) -> t (along the axis); the
    // two outer ones are the caller's loops, the third is below here.
    cell[ml->outer] = b;
    cell[ml->group] = a;
    cell[axis] = 0;

    // start of the line in memory
    size_t start = decomp_index(d, cell[0], cell[1], cell[2]);

    // start of the line in the buffers
    size_t line = (size_t)a * (size_t)length;

    /*
     * The physical term of the whole line, in a single call. Only the eta step
     * carries g, and its lines run along x: in here y, z and the time are
     * constant, the tests that govern g do not change from one cell to the
     * next and the cells are contiguous in memory. That is what g_line
     * exploits.
     */
    if (axis == 0) {
        /* Timed separately: g is the right-hand side, not the system, and as
         * long as the two costs were together inside eta_sys there was no way
         * of knowing how much was left to gain on each. Two clock readings per
         * line, that is less than half a percent of the step at the sizes that
         * matter. */
        uint64_t g_start = time_ns();

        g_line(d, mline.data, mline.state, mline.k_porosity,
               cell[1], cell[2], mline.t_step, mline.v_comp,
               ml->abscissa, source_term + line);
        mline.source_term = source_term + line;
        ml->g_ns[(size_t)workers_id() * COUNTER_STRIDE] += time_ns() - g_start;
    }

    for (int t = 0; t < length; t++) {
        cell[axis] = t;

        size_t here = start + (size_t)t * ml->step;
        size_t at = line + (size_t)t;
        MomentumRow row = momentum_row(&mline, cell, here);

        lower[at] = row.a;
        diagonal[at] = row.b;
        upper[at] = row.c;
        known[at] = row.f;
    }
}

/* Adds the write-back of ONE line to the arrival field. */
static void momentum_writeback_line(const MomentumLines *ml, int b, int a,
                                    const Real *restrict increment) {
    const Decomp *restrict d = ml->d;
    Real *restrict target = ml->target;
    const int axis = ml->axis;
    const int length = ml->length;
    int cell[3];

    cell[ml->outer] = b;
    cell[ml->group] = a;
    cell[axis] = 0;

    size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
    size_t line = (size_t)a * (size_t)length;

    for (int t = 0; t < length; t++) {
        target[start + (size_t)t * ml->step] += increment[line + (size_t)t];
    }
}

static void momentum_direction(const Decomp *d,
                               SolverMemState *solver_mem_state,
                               Data *data, int t_step, int v_comp, int axis,
                               SolverStats *solver_stats) {
    /*
     * The physics of a point is in momentum_row.h, which is shared with the
     * other backend: here the line is fixed once and then the row is requested
     * cell by cell. The row never reaches memory, the function is inline.
     */
    const MomentumLine mline =
        momentum_line(d, solver_mem_state, data, t_step, v_comp, axis);

    /*
     * The arrival stage, for writing: the write-back is added to it at the
     * end.
     */
    VectorField *to = (axis == 0) ? &solver_mem_state->eta
                    : (axis == 1) ? &solver_mem_state->zeta
                                  : &solver_mem_state->u;
    Real *target = (v_comp == 0) ? to->v_x
                 : (v_comp == 1) ? to->v_y
                                 : to->v_z;

    /*
     * The lines run along `axis`; of the other two directions, `group`
     * collects the lines solved together and is the one closest in memory,
     * `outer` is the outer loop.
     */
    const int group = (axis == 0) ? 1 : 0;
    const int outer = (axis == 2) ? 1 : 2;
    const int length = d->n[axis];
    const int lines = d->n[group];
    const size_t step = d->stride[axis];

    const int planes = d->n[outer];
    const size_t line_room = (size_t)lines * (size_t)length;

    /*
     * The planes are independent, but the work arrays are not: every thread
     * that takes one must have its own. They are allocated in a single block,
     * five per slot -- six in the eta step, where the sixth is the physical
     * term g -- and each slot belongs to one thread for the whole duration of
     * the loop.
     *
     * The sixth is indexed per line like `known`: same property, so it is safe
     * both when it is the planes that are shared out and when it is the lines.
     *
     * The planes can be shared out only if along this axis the process holds
     * the whole line: otherwise every group goes through schur_solve_mpi,
     * which communicates, and the collectives must all go in the same order on
     * all the processes. In that case the threads share out the lines inside
     * the plane, where there is no communication at all.
     */
    const bool whole_axis = (d->n[axis] == d->n_global[axis]);
    const WorkersLineSchedule schedule =
        workers_line_schedule(whole_axis, planes);
    const int slots = schedule.slots;
    const bool split_lines = schedule.split_lines;

    /*
     * The abscissae, instead, are the same for all the lines of the block --
     * along x the line is always the same row of coordinates -- so they are
     * filled once and the threads just read them.
     */
    const int arrays = (axis == 0) ? 6 : 5;
    Real *abscissa = NULL;
    const int counters = workers_available();
    uint64_t *g_ns = xmalloc((size_t)counters * COUNTER_STRIDE *
                             sizeof(uint64_t));

    for (int i = 0; i < counters * COUNTER_STRIDE; i++) {
        g_ns[i] = 0;
    }
    if (axis == 0) {
        abscissa = xmalloc((size_t)length * sizeof(Real));
        forcing_line_coords(d, v_comp, abscissa);
    }

    const MomentumLines ml = {
        .mline = &mline,
        .d = d,
        .target = target,
        .abscissa = abscissa,
        .g_ns = g_ns,
        .axis = axis,
        .group = group,
        .outer = outer,
        .length = length,
        .step = step,
    };

    Real *pool = xmalloc((size_t)slots * (size_t)arrays * line_room *
                         sizeof(Real));

    if (split_lines) {
        /*
         * Axis divided among processes. The planes must go one after the
         * other, because each goes through a collective and the collectives go
         * in the same order on all the processes; what is shared out is the
         * lines inside the plane, where there is no communication at all.
         *
         * The team is opened ONCE, here outside the loop over the planes.
         * Before, one was opened for each of the two inner loops, that is two
         * per plane: at 256^3 with four processes that made about 4100
         * openings per time step, and that was the dominant cost. With fixed
         * ranks, with identical MPI work row by row, the step went from 1295
         * ms with one thread to 5094 with fourteen.
         *
         * Three barriers per plane remain, but a barrier inside a team that is
         * already alive is a different thing from creating and destroying it.
         */
        Real *restrict lower = pool;
        Real *restrict diagonal = pool + line_room;
        Real *restrict upper = pool + 2 * line_room;
        Real *restrict known = pool + 3 * line_room;
        Real *restrict increment = pool + 4 * line_room;
        Real *restrict source_term = (axis == 0) ? pool + 5 * line_room
                                                 : NULL;

        WORKERS_PARALLEL(true)
        {
            for (int b = 0; b < planes; b++) {
                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    momentum_assemble_line(&ml, b, a,
                                           lower, diagonal, upper, known,
                                           source_term);
                }
                /* Implicit barrier of WORKERS_FOR: the system is complete. */

                WORKERS_MASTER
                schur_solve_mpi(axis, lines, length,
                                lower, diagonal, upper, known, increment);

                /* WORKERS_MASTER has no barrier of its own: without this the
                 * other threads would read `increment' while the master is
                 * still writing it. */
                WORKERS_BARRIER

                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    momentum_writeback_line(&ml, b, a, increment);
                }
            }
        }
    } else {
        /*
         * Axis entirely local, or a single thread: the planes are independent
         * and are shared out directly, one team for the whole direction. Here
         * every thread needs its own work arrays, and that is why the pool has
         * more than one slot.
         */
        WORKERS_PARALLEL_FOR(slots > 1)
        for (int b = 0; b < planes; b++) {
            Real *slot = pool + (size_t)workers_slot(slots) *
                                (size_t)arrays * line_room;
            Real *restrict lower = slot;
            Real *restrict diagonal = slot + line_room;
            Real *restrict upper = slot + 2 * line_room;
            Real *restrict known = slot + 3 * line_room;
            Real *restrict increment = slot + 4 * line_room;
            Real *restrict source_term = (axis == 0) ? slot + 5 * line_room
                                                     : NULL;

            for (int a = 0; a < lines; a++) {
                momentum_assemble_line(&ml, b, a,
                                       lower, diagonal, upper, known,
                                       source_term);
            }

            schur_solve_mpi(axis, lines, length,
                            lower, diagonal, upper, known, increment);

            for (int a = 0; a < lines; a++) {
                momentum_writeback_line(&ml, b, a, increment);
            }
        }
    }

    /*
     * The longest branch, not the sum: the threads work together, so the
     * wall-clock time spent on g is that of the thread that did the most of
     * it. That is how the number stays comparable with eta_sys, which is
     * wall-clock.
     */
    uint64_t slowest = 0;
    for (int i = 0; i < counters; i++) {
        uint64_t thread_ns = g_ns[(size_t)i * COUNTER_STRIDE];

        if (thread_ns > slowest) {
            slowest = thread_ns;
        }
    }
    solver_stats->momentum_source += slowest;

    free(g_ns);
    free(abscissa);
    free(pool);
}

void momentum_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   Data *data, int t_step, SolverStats *solver_stats) {
    SchurBackend *backend = solver_mem_state->backend;
    Real *restrict rhs = backend->rhs;
    Real *restrict tmp = backend->tmp;

    (void)rhs;
    (void)tmp;

    /*
     * The vectorised versions solve the whole line in one go, so they are
     * valid as long as that axis is not divided among several processes.
     */
    uint64_t start_ns = time_ns();
    for (int v_comp = 0; v_comp < 3; v_comp++) {
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp,
                           0, solver_stats);
    }
    solver_stats->eta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    for (int v_comp = 0; v_comp < 3; v_comp++) {
#if defined(USE_SIMD) && SIMD_AVAILABLE
        // this condition in the if means that the y axis is not divided among
        // several processes, so the SIMD version can be used
        if (decomp->n[1] == decomp->n_global[1]) {
            update_zeta_simd(decomp, solver_mem_state, rhs, tmp, data, t_step,
                             v_comp, ZETA_SIMD_LINES);
            continue;
        }
#endif
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp,
                           1, solver_stats);
    }
    solver_stats->zeta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    for (int v_comp = 0; v_comp < 3; v_comp++) {
#if defined(USE_SIMD) && SIMD_AVAILABLE
        if (decomp->n[2] == decomp->n_global[2]) {
            update_u_simd(decomp, solver_mem_state, rhs, tmp, data, t_step,
                          v_comp, U_SIMD_LINES);
            continue;
        }
#endif
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp,
                           2, solver_stats);
    }
    solver_stats->u_sys += time_ns() - start_ns;
}
