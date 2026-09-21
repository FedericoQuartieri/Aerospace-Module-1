#include "pressure.h"
#include "pressure_common.h"
#include "backend.h"
#include "backend_schur.h"
#include "schur.h"
#include "utils.h"
#include "workers.h"

/*
 * The preprocessing of the three axes (Lecture 5, p. 32, point 1): none of the
 * three matrices depends on the time step, so it is prepared once at start-up
 * and during the simulation only the right-hand side remains to be written.
 */
void pressure_plans_init(const Decomp *d, SchurPlan plan[3]) {
    for (int axis = 0; axis < 3; axis++) {
        size_t room = (size_t)d->n[axis] * sizeof(Real);
        Real *a = xmalloc(room);
        Real *b = xmalloc(room);
        Real *c = xmalloc(room);

        pressure_matrix(d, axis, a, b, c);
        schur_plan_init(&plan[axis], axis, d->n[axis], a, b, c);

        free(c);
        free(b);
        free(a);
    }
}

void pressure_plans_free(SchurPlan plan[3]) {
    for (int axis = 0; axis < 3; axis++) {
        schur_plan_free(&plan[axis]);
    }
}

/*
 * One step of the pressure cascade (Lecture 5, p. 7):
 *
 *   (I - d_xx) psi = -div(u)/dt,  then  (I - d_yy) phi = psi,
 *                                 then  (I - d_zz) fi  = phi
 *
 * The three steps have the same matrix: only the axis, the field it reads from
 * and the one it writes to change. This function does one, and it is called
 * three times. The matrix is already in `plan`, here the right-hand side is
 * written and the answer is collected.
 */
/*
 * The context of a pressure line. Same reason as its twin in momentum.c: the
 * body is walked by two different loop structures, and repeating it in two
 * places would be the easiest way to make them diverge.
 */
typedef struct {
    const Decomp *d;
    const Real *source;
    Real *target;
    int axis;
    int group;
    int outer;
    int length;
    size_t step;
} PressureLines;

/*
 * Collects the right-hand side of ONE line, from the field to the working
 * form.
 */
static void pressure_gather_line(const PressureLines *pl, int b, int a,
                                 Real *restrict known) {
    const Real *restrict source = pl->source;
    const int length = pl->length;
    int cell[3];

    cell[pl->outer] = b;
    cell[pl->group] = a;
    cell[pl->axis] = 0;

    size_t start = decomp_index(pl->d, cell[0], cell[1], cell[2]);
    size_t line = (size_t)a * (size_t)length;

    for (int t = 0; t < length; t++) {
        known[line + (size_t)t] = source[start + (size_t)t * pl->step];
    }
}

/* Writes the solution of ONE line back into the arrival field. */
static void pressure_scatter_line(const PressureLines *pl, int b, int a,
                                  const Real *restrict answer) {
    Real *restrict target = pl->target;
    const int length = pl->length;
    int cell[3];

    cell[pl->outer] = b;
    cell[pl->group] = a;
    cell[pl->axis] = 0;

    size_t start = decomp_index(pl->d, cell[0], cell[1], cell[2]);
    size_t line = (size_t)a * (size_t)length;

    for (int t = 0; t < length; t++) {
        target[start + (size_t)t * pl->step] = answer[line + (size_t)t];
    }
}

static void pressure_direction(const Decomp *restrict d,
                               const SchurPlan *plan,
                               const Real *restrict source,
                               Real *restrict target) {
    const int axis = plan->axis;
    const int group = (axis == 0) ? 1 : 0;
    const int outer = (axis == 2) ? 1 : 2;
    const int length = d->n[axis];
    const int lines = d->n[group];
    const size_t step = d->stride[axis];

    const int planes = d->n[outer];
    const size_t line_room = (size_t)lines * (size_t)length;

    /* Same distribution as momentum: one plane per thread as long as the axis
     * is all here, the lines of the plane when there is communication. */
    const bool whole_axis = (d->n[axis] == d->n_global[axis]);
    const WorkersLineSchedule schedule =
        workers_line_schedule(whole_axis, planes);
    const int slots = schedule.slots;
    const bool split_lines = schedule.split_lines;

    const PressureLines pl = {
        .d = d,
        .source = source,
        .target = target,
        .axis = axis,
        .group = group,
        .outer = outer,
        .length = length,
        .step = step,
    };

    Real *pool = xmalloc((size_t)slots * 2 * line_room * sizeof(Real));

    if (split_lines) {
        /* A single team for the whole direction instead of two per plane: the
         * same fix as momentum.c, for the same measured reason. */
        Real *restrict known = pool;
        Real *restrict answer = pool + line_room;

        WORKERS_PARALLEL(true)
        {
            for (int b = 0; b < planes; b++) {
                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    pressure_gather_line(&pl, b, a, known);
                }

                WORKERS_MASTER
                schur_plan_solve(plan, lines, known, answer);

                /* WORKERS_MASTER has no barrier of its own. */
                WORKERS_BARRIER

                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    pressure_scatter_line(&pl, b, a, answer);
                }
            }
        }
    } else {
        WORKERS_PARALLEL_FOR(slots > 1)
        for (int b = 0; b < planes; b++) {
            Real *slot = pool + (size_t)workers_slot(slots) * 2 * line_room;
            Real *restrict known = slot;
            Real *restrict answer = slot + line_room;

            for (int a = 0; a < lines; a++) {
                pressure_gather_line(&pl, b, a, known);
            }

            schur_plan_solve(plan, lines, known, answer);

            for (int a = 0; a < lines; a++) {
                pressure_scatter_line(&pl, b, a, answer);
            }
        }
    }

    free(pool);
}

void pressure_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer,
                   SolverStats *solver_stats)
{
    const SchurBackend *backend = solver_mem_state->backend;
    const SchurPlan *plan = backend->pressure_plan;

    Real *buffer = pressure_buffer->v;
    Real *star = solver_mem_state->pressure_star.v;

    /* The two fields swap roles at every step, so two are enough. */
    uint64_t start_ns = time_ns();
    compute_div(decomp, buffer, &solver_mem_state->u);
    pressure_direction(decomp, &plan[0], buffer, star);   /* psi */
    solver_stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(decomp, &plan[1], star, buffer);   /* phi */
    solver_stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    pressure_direction(decomp, &plan[2], buffer, star);   /* fi  */
    solver_stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(decomp, solver_mem_state);
    solver_stats->pressure_update += time_ns() - start_ns;
}
