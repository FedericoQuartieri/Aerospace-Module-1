#include "momentum.h"
#include "momentum_row.h"
#include "schur.h"
#include "utils.h"
#include "workers.h"

/*
 * I tre passi della quantita' di moto sono la stessa operazione ripetuta su
 * assi diversi (Lecture 5, p. 6):
 *
 *   (I - g d_xx)(eta^{n+1}  - eta^n)  = xi^{n+1}  - eta^n
 *   (I - g d_yy)(zeta^{n+1} - zeta^n) = eta^{n+1} - zeta^n
 *   (I - g d_zz)(u^{n+1}    - u^n)    = zeta^{n+1} - u^n
 *
 * Cambiano l'asse, il campo di partenza e quello di arrivo; il resto e'
 * identico. Questa funzione li fa tutti e tre, e la si chiama con l'asse.
 *
 * Il sistema di ogni linea viene scritto per esteso nelle tre diagonali e nel
 * termine noto, e passato a schur_solve_mpi: se lungo quell'asse c'e' un solo
 * processo lui usa Thomas normale, altrimenti ricuce i pezzi delle linee
 * spezzate. Le linee di uno stesso gruppo vengono risolte insieme, cosi' il
 * gruppo costa una comunicazione invece di una per linea.
 */
static void momentum_direction(const Decomp *d,
                               SolverMemState *solver_mem_state,
                               Data *data, int t_step, int v_comp, int axis) {
    /*
     * La fisica di un punto sta in momentum_row.h, che e' condiviso con
     * l'altro backend: qui si fissa la linea una volta e poi si chiede la
     * riga cella per cella.  La riga non arriva mai in memoria, la funzione
     * e' inline.
     */
    const MomentumLine mline =
        momentum_line(d, solver_mem_state, data, t_step, v_comp, axis);

    /* Lo stadio di arrivo, in scrittura: il riporto ci si somma alla fine. */
    VectorField *to = (axis == 0) ? &solver_mem_state->eta
                    : (axis == 1) ? &solver_mem_state->zeta
                                  : &solver_mem_state->u;
    Real *target = (v_comp == 0) ? to->v_x
                 : (v_comp == 1) ? to->v_y
                                 : to->v_z;

    /*
     * Le linee corrono lungo `axis`; delle altre due direzioni, `group`
     * raccoglie le linee risolte insieme ed e' quella piu' vicina in memoria,
     * `outer` e' il ciclo esterno.
     */
    const int group = (axis == 0) ? 1 : 0;
    const int outer = (axis == 2) ? 1 : 2;
    const int length = d->n[axis];
    const int lines = d->n[group];
    const size_t step = d->stride[axis];

    const int planes = d->n[outer];
    const size_t line_room = (size_t)lines * (size_t)length;

    /*
     * I piani sono indipendenti, ma i cinque array di lavoro no: ogni thread
     * che ne prende uno deve avere i propri.  Sono allocati in un blocco solo,
     * cinque per slot, e ogni slot appartiene a un thread per tutta la durata
     * del ciclo.
     *
     * I piani si possono spartire solo se lungo questo asse il processo tiene
     * tutta la linea: altrimenti ogni gruppo passa da schur_solve_mpi, che
     * comunica, e le collettive vanno tutte nello stesso ordine su tutti i
     * processi.  In quel caso i thread si spartiscono le linee dentro il
     * piano, dove non si comunica affatto.
     */
    const bool whole_axis = (d->n[axis] == d->n_global[axis]);
    const int slots = workers_slots(whole_axis, planes);
    const bool split_lines = (slots < 2) && workers_many();

    Real *pool = xmalloc((size_t)slots * 5 * line_room * sizeof(Real));

    WORKERS_PARALLEL_FOR(slots > 1)
    for (int b = 0; b < planes; b++) {
        Real *slot = pool + (size_t)workers_slot(slots) * 5 * line_room;
        Real *restrict lower = slot;
        Real *restrict diagonal = slot + line_room;
        Real *restrict upper = slot + 2 * line_room;
        Real *restrict known = slot + 3 * line_room;
        Real *restrict increment = slot + 4 * line_room;

        WORKERS_PARALLEL_FOR(split_lines)
        for (int a = 0; a < lines; a++) {
            int cell[3];

            cell[outer] = b;
            cell[group] = a;
            cell[axis] = 0;

            size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
            size_t line = (size_t)a * (size_t)length;

            for (int t = 0; t < length; t++) {
                cell[axis] = t;

                size_t here = start + (size_t)t * step;
                size_t at = line + (size_t)t;
                MomentumRow row = momentum_row(&mline, cell, here);

                lower[at] = row.a;
                diagonal[at] = row.b;
                upper[at] = row.c;
                known[at] = row.f;
            }
        }

        schur_solve_mpi(axis, lines, length,
                        lower, diagonal, upper, known, increment);

        WORKERS_PARALLEL_FOR(split_lines)
        for (int a = 0; a < lines; a++) {
            int cell[3];

            cell[outer] = b;
            cell[group] = a;
            cell[axis] = 0;

            size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
            size_t line = (size_t)a * (size_t)length;

            for (int t = 0; t < length; t++) {
                target[start + (size_t)t * step] += increment[line + (size_t)t];
            }
        }
    }

    free(pool);
}

void momentum_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data, int t_step, SolverStats *solver_stats) {
    (void)rhs;
    (void)tmp;

    /*
     * Le versioni vettorizzate risolvono la linea intera in un colpo solo,
     * quindi valgono finche' quell'asse non e' diviso fra piu' processi.
     */
    uint64_t start_ns = time_ns();
    for (int v_comp = 0; v_comp < 3; v_comp++) {
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp, 0);
    }
    solver_stats->eta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    for (int v_comp = 0; v_comp < 3; v_comp++) {
#if defined(USE_SIMD) && SIMD_AVAILABLE
        if (decomp->n[1] == decomp->n_global[1]) {
            update_zeta_simd(decomp, solver_mem_state, rhs, tmp, data, t_step,
                             v_comp, ZETA_SIMD_LINES);
            continue;
        }
#endif
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp, 1);
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
        momentum_direction(decomp, solver_mem_state, data, t_step, v_comp, 2);
    }
    solver_stats->u_sys += time_ns() - start_ns;
}
