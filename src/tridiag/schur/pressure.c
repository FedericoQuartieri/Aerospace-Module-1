#include "pressure.h"
#include "pressure_common.h"
#include "backend.h"
#include "backend_schur.h"
#include "schur.h"
#include "utils.h"
#include "workers.h"

/*
 * Il preprocessing dei tre assi (Lecture 5, p. 32, punto 1): nessuna delle tre
 * matrici dipende dal passo temporale, quindi si prepara una volta all'avvio e
 * durante la simulazione resta da scrivere solo il termine noto.
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
 * Un passo della cascata della pressione (Lecture 5, p. 7):
 *
 *   (I - d_xx) psi = -div(u)/dt,  poi  (I - d_yy) phi = psi,
 *                                 poi  (I - d_zz) fi  = phi
 *
 * I tre passi hanno la stessa matrice: cambiano solo l'asse, il campo da cui
 * si legge e quello in cui si scrive. Questa funzione ne fa uno, e la si
 * chiama tre volte. La matrice sta gia' in `plan`, qui si scrive il termine
 * noto e si raccoglie la risposta.
 */
/*
 * Il contesto di una linea di pressione. Stessa ragione della gemella in
 * momentum.c: il corpo e' percorso da due strutture di cicli diverse, e
 * ripeterlo in due punti sarebbe il modo piu' facile per farle divergere.
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

/* Raccoglie il termine noto di UNA linea, dal campo alla forma di lavoro. */
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

/* Riporta la soluzione di UNA linea nel campo di arrivo. */
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

    /* Stessa spartizione della quantita' di moto: un piano per thread finche'
     * l'asse e' tutto qui, le linee del piano quando invece si comunica. */
    const bool whole_axis = (d->n[axis] == d->n_global[axis]);
    const int slots = workers_slots(whole_axis, planes);
    const bool split_lines = (slots < 2) && workers_many();

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
        /* Un team solo per l'intera direzione invece di due per piano: la
         * stessa correzione di momentum.c, per la stessa ragione misurata. */
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

                /* WORKERS_MASTER non ha barriera propria. */
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

    /* I due campi si scambiano il ruolo a ogni passo, cosi' ne bastano due. */
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
