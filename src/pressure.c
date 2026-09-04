#include "pressure.h"
#include "schur.h"
#include "utils.h"
#include "workers.h"

/*
 * Divergenza della velocita', divisa per il passo temporale: e' il termine
 * noto del primo dei tre passi della pressione.
 *
 * Sulle tre facce inferiori del dominio vale zero (la velocita' e' a
 * divergenza nulla). Sono facce globali: un processo che non le tocca calcola
 * la divergenza anche li', leggendo la cella precedente dall'anello di
 * contorno.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u) {
    const Real *restrict u_x = u->v_x;
    const Real *restrict u_y = u->v_y;
    const Real *restrict u_z = u->v_z;
    const size_t stride_y = d->stride[1];
    const size_t stride_z = d->stride[2];
    const Real x_factor = -(Real)DX_INVERSE / (Real)DT;
    const Real y_factor = -(Real)DY_INVERSE / (Real)DT;
    const Real z_factor = -(Real)DZ_INVERSE / (Real)DT;

    WORKERS_PARALLEL_FOR(workers_many())
    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            if (gk == 0 || gj == 0) {
                for (int i = 0; i < d->n[0]; i++) {
                    u_div[row + (size_t)i] = (Real)0;
                }
                continue;
            }

            int i0 = d->is_first[0] ? 1 : 0;
            if (d->is_first[0]) {
                u_div[row] = (Real)0;
            }

            for (int i = i0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;

                u_div[index] =
                    (u_x[index] - u_x[index - 1]) * x_factor +
                    (u_y[index] - u_y[index - stride_y]) * y_factor +
                    (u_z[index] - u_z[index - stride_z]) * z_factor;
            }
        }
    }
}

/*
 * La matrice di una linea della cascata di pressione.
 *
 * E' la stessa per tutte le linee dell'asse e non cambia mai nel tempo: ogni
 * riga dipende solo dalla posizione globale del suo punto.
 *
 * Le righe agli estremi sono quelle della condizione di Neumann omogenea
 * (Lecture 5, pp. 15-16), e spettano solo a chi tocca davvero la parete: un
 * processo in mezzo al dominio usa ovunque la riga interna. Sono asimmetriche
 * perche' lo e' la discretizzazione: a sinistra il nodo fantasma dista due
 * mezze celle, a destra una.
 */
static void pressure_matrix(const Decomp *restrict d, int axis,
                            Real *restrict a,
                            Real *restrict b,
                            Real *restrict c) {
    const int length = d->n[axis];
    const int last_global = d->n_global[axis] - 1;
    const Real w = (axis == 0) ? -(Real)DX_INVERSE_SQUARE
                 : (axis == 1) ? -(Real)DY_INVERSE_SQUARE
                               : -(Real)DZ_INVERSE_SQUARE;

    for (int t = 0; t < length; t++) {
        int along = decomp_global(d, t, axis);

        if (along == 0) {
            a[t] = (Real)0;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = (Real)2 * w;
        } else if (along == last_global) {
            a[t] = w;
            b[t] = (Real)1 - w;
            c[t] = (Real)0;
        } else {
            a[t] = w;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = w;
        }
    }
}

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

static void update_pressure(const Decomp *restrict d,
                            SolverMemState *restrict solver_mem_state) {
    Real *restrict pressure = solver_mem_state->pressure.v;
    Real *restrict phi_high = solver_mem_state->pressure_star.v;

    WORKERS_PARALLEL_FOR(workers_many())
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                Real phi = phi_high[index];
                Real pressure_new = pressure[index] + phi;

                pressure[index] = pressure_new;
                phi_high[index] = pressure_new + phi;
            }
        }
    }
}

void pressure_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   const SchurPlan plan[3],
                   ScalarField *pressure_buffer,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   SolverStats *solver_stats)
{
    (void)rhs;
    (void)tmp;

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
