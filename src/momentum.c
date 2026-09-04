#include "momentum.h"
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
/*
 * Il contesto di una linea: quello che il corpo del ciclo deve sapere e che
 * non cambia da una linea all'altra.
 *
 * Serve perche' lo stesso corpo e' ora percorso da due strutture di cicli
 * diverse -- una che sparisce i piani fra i thread, una che sparisce le linee
 * dentro il piano -- e ripetere quindici argomenti in due punti sarebbe il
 * modo piu' facile per farle divergere senza accorgersene.
 */
typedef struct {
    const Decomp *d;
    SolverMemState *solver_mem_state;
    Data *data;
    const Real *k_porosity;
    const Real *source;
    Real *target;
    Real inverse_square;
    bool same_direction;
    int t_step;
    int v_comp;
    int axis;
    int group;
    int outer;
    int length;
    int last_global;
    size_t step;
} MomentumLines;

/*
 * Scrive nelle tre diagonali e nel termine noto il sistema di UNA linea.
 *
 * I puntatori caldi vengono ripresi in locale con `restrict': attraverso la
 * struttura il compilatore perderebbe la garanzia di non aliasing che aveva
 * quando il corpo stava dentro al ciclo, e con essa la vettorizzazione.
 */
static void momentum_assemble_line(const MomentumLines *ml, int b, int a,
                                   Real *restrict lower,
                                   Real *restrict diagonal,
                                   Real *restrict upper,
                                   Real *restrict known) {
    const Decomp *restrict d = ml->d;
    const Real *restrict k_porosity = ml->k_porosity;
    const Real *restrict source = ml->source;
    const Real *restrict target = ml->target;
    const int axis = ml->axis;
    const int length = ml->length;
    int cell[3];

    cell[ml->outer] = b;
    cell[ml->group] = a;
    cell[axis] = 0;

    size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
    size_t line = (size_t)a * (size_t)length;

    for (int t = 0; t < length; t++) {
        cell[axis] = t;

        int gi = decomp_global(d, cell[0], 0);
        int gj = decomp_global(d, cell[1], 1);
        int gk = decomp_global(d, cell[2], 2);
        int along = decomp_global(d, t, axis);

        size_t here = start + (size_t)t * ml->step;
        size_t at = line + (size_t)t;
        Real k_i = k_porosity[here];
        Real w_i = -gamma_from_k(k_i) * ml->inverse_square;

        if (along == 0) {
            /* Parete inferiore del dominio: valore imposto. */
            lower[at] = 0.0;
            diagonal[at] = 1.0;
            upper[at] = 0.0;
            known[at] = bc_left(ml->data->bc_velocity, gi, gj, gk,
                                ml->t_step, ml->v_comp);
            continue;
        }

        Real rhs = source[here] - target[here];

        /* Solo il primo passo porta il termine fisico g. */
        if (axis == 0) {
            rhs += (DT / beta_from_k(k_i)) *
                   g_value(d, cell[0], cell[1], cell[2], ml->t_step, k_i,
                           ml->solver_mem_state, ml->data, ml->v_comp);
        }

        if (along < ml->last_global) {
            lower[at] = w_i;
            diagonal[at] = 1.0 - 2.0 * w_i;
            upper[at] = w_i;
            known[at] = rhs;
        } else if (ml->same_direction) {
            lower[at] = 0.0;
            diagonal[at] = 1.0;
            upper[at] = 0.0;
            known[at] = bc_right(ml->data->bc_velocity, gi, gj, gk,
                                 ml->t_step, ml->v_comp);
        } else {
            /* Parete superiore, componente tangente: nodo fantasma
             * eliminato con la condizione al contorno. */
            Real right_value = bc_right(ml->data->bc_velocity, gi, gj, gk,
                                        ml->t_step, ml->v_comp);
            lower[at] = w_i;
            diagonal[at] = 1.0 - 3.0 * w_i;
            upper[at] = 0.0;
            known[at] = rhs - 2.0 * w_i * right_value;
        }
    }
}

/* Somma al campo di arrivo il riporto di UNA linea. */
static void momentum_writeback_line(const MomentumLines *ml, int b, int a,
                                    const Real *restrict increment) {
    const Decomp *restrict d = ml->d;
    Real *restrict target = ml->target;
    const int length = ml->length;
    int cell[3];

    cell[ml->outer] = b;
    cell[ml->group] = a;
    cell[ml->axis] = 0;

    size_t start = decomp_index(d, cell[0], cell[1], cell[2]);
    size_t line = (size_t)a * (size_t)length;

    for (int t = 0; t < length; t++) {
        target[start + (size_t)t * ml->step] += increment[line + (size_t)t];
    }
}

static void momentum_direction(const Decomp *d,
                               SolverMemState *solver_mem_state,
                               Data *data, int t_step, int v_comp, int axis) {
    const Real *restrict k_porosity;
    const Real *restrict source;
    Real *restrict target;

    /* eta parte da u, zeta parte da eta, u parte da zeta. */
    const VectorField *from = (axis == 0) ? &solver_mem_state->u
                            : (axis == 1) ? &solver_mem_state->eta
                                          : &solver_mem_state->zeta;
    VectorField *to = (axis == 0) ? &solver_mem_state->eta
                    : (axis == 1) ? &solver_mem_state->zeta
                                  : &solver_mem_state->u;

    switch (v_comp) {
        case 0:
            k_porosity = solver_mem_state->k.v_x;
            source = from->v_x;
            target = to->v_x;
            break;
        case 1:
            k_porosity = solver_mem_state->k.v_y;
            source = from->v_y;
            target = to->v_y;
            break;
        case 2:
            k_porosity = solver_mem_state->k.v_z;
            source = from->v_z;
            target = to->v_z;
            break;
        default:
            fprintf(stderr, "Value of v_comp doesn't exist");
            exit(1);
    }

    const Real inverse_square = (axis == 0) ? (Real)DX_INVERSE_SQUARE
                              : (axis == 1) ? (Real)DY_INVERSE_SQUARE
                                            : (Real)DZ_INVERSE_SQUARE;
    /* Componente normale alla parete: il valore al bordo e' imposto. */
    const bool same_direction = (v_comp == axis);

    /*
     * Le linee corrono lungo `axis`; delle altre due direzioni, `group`
     * raccoglie le linee risolte insieme ed e' quella piu' vicina in memoria,
     * `outer` e' il ciclo esterno.
     */
    const int group = (axis == 0) ? 1 : 0;
    const int outer = (axis == 2) ? 1 : 2;
    const int length = d->n[axis];
    const int lines = d->n[group];
    const int last_global = d->n_global[axis] - 1;
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
    const WorkersLineSchedule schedule =
        workers_line_schedule(whole_axis, planes);
    const int slots = schedule.slots;
    const bool split_lines = schedule.split_lines;

    const MomentumLines ml = {
        .d = d,
        .solver_mem_state = solver_mem_state,
        .data = data,
        .k_porosity = k_porosity,
        .source = source,
        .target = target,
        .inverse_square = inverse_square,
        .same_direction = same_direction,
        .t_step = t_step,
        .v_comp = v_comp,
        .axis = axis,
        .group = group,
        .outer = outer,
        .length = length,
        .last_global = last_global,
        .step = step,
    };

    Real *pool = xmalloc((size_t)slots * 5 * line_room * sizeof(Real));

    if (split_lines) {
        /*
         * Asse diviso fra processi. I piani devono andare in fila, perche'
         * ognuno passa da una collettiva e le collettive vanno nello stesso
         * ordine su tutti i processi; a spartirsi sono le linee dentro il
         * piano, dove non si comunica affatto.
         *
         * Il team si apre UNA volta, qui fuori dal ciclo sui piani. Prima se
         * ne apriva uno per ciascuno dei due cicli interni, cioe' due per
         * piano: a 256^3 con quattro processi facevano circa 4100 aperture
         * per passo temporale, ed erano quelle il costo dominante.
         * A rank fissi, con lavoro MPI identico riga per riga, il
         * passo andava da 1295 ms con un thread a 5094 con quattordici.
         *
         * Restano tre barriere per piano, ma una barriera dentro un team gia'
         * vivo e' un'altra cosa dal crearlo e distruggerlo.
         */
        Real *restrict lower = pool;
        Real *restrict diagonal = pool + line_room;
        Real *restrict upper = pool + 2 * line_room;
        Real *restrict known = pool + 3 * line_room;
        Real *restrict increment = pool + 4 * line_room;

        WORKERS_PARALLEL(true)
        {
            for (int b = 0; b < planes; b++) {
                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    momentum_assemble_line(&ml, b, a,
                                           lower, diagonal, upper, known);
                }
                /* Barriera implicita di WORKERS_FOR: il sistema e' completo. */

                WORKERS_MASTER
                schur_solve_mpi(axis, lines, length,
                                lower, diagonal, upper, known, increment);

                /* WORKERS_MASTER non ha barriera propria: senza questa gli
                 * altri thread leggerebbero `increment' mentre il master lo
                 * sta ancora scrivendo. */
                WORKERS_BARRIER

                WORKERS_FOR
                for (int a = 0; a < lines; a++) {
                    momentum_writeback_line(&ml, b, a, increment);
                }
            }
        }
    } else {
        /*
         * Asse tutto locale, oppure un thread solo: i piani sono indipendenti
         * e si spartiscono direttamente, un team per l'intera direzione. Qui
         * ogni thread ha bisogno dei propri array di lavoro, ed e' per questo
         * che il pool ha piu' di uno slot. Questa strada non cambia.
         */
        WORKERS_PARALLEL_FOR(slots > 1)
        for (int b = 0; b < planes; b++) {
            Real *slot = pool + (size_t)workers_slot(slots) * 5 * line_room;
            Real *restrict lower = slot;
            Real *restrict diagonal = slot + line_room;
            Real *restrict upper = slot + 2 * line_room;
            Real *restrict known = slot + 3 * line_room;
            Real *restrict increment = slot + 4 * line_room;

            for (int a = 0; a < lines; a++) {
                momentum_assemble_line(&ml, b, a,
                                       lower, diagonal, upper, known);
            }

            schur_solve_mpi(axis, lines, length,
                            lower, diagonal, upper, known, increment);

            for (int a = 0; a < lines; a++) {
                momentum_writeback_line(&ml, b, a, increment);
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
