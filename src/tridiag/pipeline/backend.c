#include "backend.h"
#include "backend_pipeline.h"
#include "parallel.h"
#include "params.h"
#include "pressure_common.h"
#include "utils.h"
#include "workers.h"

#include <stdint.h>
#include <stdlib.h>

static void pipeline_size_overflow(const char *what) {
    if (par_rank() == 0) {
        fprintf(stderr, "pipeline: size too large for %s\n", what);
    }
    par_abort(1);
}

static size_t checked_mul(size_t a, size_t b, const char *what) {
    if (a != 0 && b > SIZE_MAX / a) {
        pipeline_size_overflow(what);
    }

    return a * b;
}

static Real *xmalloc_real_array(size_t count, const char *what) {
    return xmalloc(checked_mul(count, sizeof(Real), what));
}

/*
 * Quanti elementi di c'/d' servono per una componente lungo `axis`.
 *
 * I batch sono tutti larghi batch_lines anche quando l'ultimo e' parziale:
 * arrotondare per eccesso costa qualche riga di scratch e in cambio rende
 * l'indirizzamento una moltiplicazione invece di una somma di prefissi.
 */
static size_t axis_capacity(const Decomp *d, int axis, int batch_lines) {
    size_t line_count = pipeline_line_count(d, axis);
    size_t batch_count =
        (line_count - 1) / (size_t)batch_lines + 1;
    size_t padded_lines =
        checked_mul(batch_count, (size_t)batch_lines, "batch of lines");

    return checked_mul(padded_lines, (size_t)d->n[axis], "per-axis scratch");
}

#if !defined(PIPELINE_BATCH_LINES)
/*
 * Linee per batch dai thread del processo.
 *
 * Il batch fa due mestieri. Fra processi e' l'unita' dei messaggi alle
 * giunzioni, e un batch piccolo fa partire prima il processo successivo.
 * Dentro un processo decide quanto lavora ogni thread fra due barriere: con
 * B linee e T thread ciascuno elimina circa B/T celle per livello e poi
 * aspetta gli altri. Un valore fisso non serve i due casi: il 64 di prima
 * restava entro il 4-14% dal migliore senza thread, e con molti thread
 * arrivava a decine di volte.
 *
 * La regola -- la potenza di due piu' vicina, in scala logaritmica, a 256
 * volte i thread, al piu' 4096 -- viene dalla scansione del batch della
 * campagna di misura: sulle 32 curve perde lo 0.5% in mediana e l'11% nel
 * caso peggiore. Oltre 4096 non ci sono misure, e la regola non estrapola.
 */
static int batch_lines_for_threads(int threads) {
    long long target = 256LL * (threads > 0 ? threads : 1);
    long long power = 1;

    while (power * 2 <= target) {
        power *= 2;
    }
    if (target * target > 2 * power * power) {
        power *= 2;
    }
    return (int)(power < 4096 ? power : 4096);
}
#endif

/*
 * Chi decide, dal piu' forte: PIPELINE_BATCH_LINES a compilazione -- la
 * scansione del batch e check_pipeline.sh devono misurare esattamente il
 * valore del binario --, poi pipeline_batch_lines nel file di
 * configurazione, poi la regola.
 */
static int requested_batch_lines(void) {
#if defined(PIPELINE_BATCH_LINES)
    if (sim.pipeline_batch_lines > 0 &&
        sim.pipeline_batch_lines != PIPELINE_BATCH_LINES && par_rank() == 0) {
        fprintf(stderr,
                "pipeline: this build fixes the batch at %d lines, "
                "pipeline_batch_lines = %d is ignored\n",
                PIPELINE_BATCH_LINES, sim.pipeline_batch_lines);
    }
    return PIPELINE_BATCH_LINES;
#else
    if (sim.pipeline_batch_lines > 0) {
        return sim.pipeline_batch_lines;
    }
    return batch_lines_for_threads(workers_available());
#endif
}

static int chosen_batch_lines;

int backend_batch_lines(void) {
    return chosen_batch_lines;
}

void backend_init(const Decomp *d, SolverMemState *solver_mem_state) {
    PipelineBackend *backend = xmalloc(sizeof(PipelineBackend));
    size_t capacity = 0;

    /*
     * Lo stesso valore su tutti i processi: i messaggi alle giunzioni
     * viaggiano un batch alla volta, e due vicini con batch diversi non si
     * capirebbero. Con thread diversi fra i processi vince il piu' grande.
     */
    backend->batch_lines = (int)par_max_long(requested_batch_lines());
    if (backend->batch_lines < 1) {
        backend->batch_lines = 1;
    }
    chosen_batch_lines = backend->batch_lines;

    for (int axis = 0; axis < 3; axis++) {
        size_t needed = axis_capacity(d, axis, backend->batch_lines);

        if (needed > capacity) {
            capacity = needed;
        }
    }

    /*
     * Il prezzo della pipeline: per poter fare la sostituzione all'indietro
     * piu' tardi bisogna tenere c' e d' di tutto il blocco locale, per tutte
     * e tre le componenti.  Su 128^3 in double sono circa 200 MB oltre allo
     * stato del solutore.  Schur non li paga, ma paga tre risoluzioni locali
     * per linea invece di una.
     */
    backend->component_capacity = capacity;
    backend->c_prime =
        xmalloc_real_array(checked_mul(3, capacity, "c'"), "c'");
    backend->d_prime =
        xmalloc_real_array(checked_mul(3, capacity, "d'"), "d'");
    backend->forward =
        xmalloc_real_array(checked_mul(2,
                                       (size_t)backend->batch_lines,
                                       "forward junctions"),
                           "forward junctions");
    backend->backward =
        xmalloc_real_array((size_t)backend->batch_lines,
                           "backward junctions");

    /*
     * Il g di un batch di linee lungo x.  Costa batch_lines * n[0] elementi --
     * su 128^3 col batch di default sono 64 KB in double, nulla accanto ai c'
     * e d' qui sopra -- e in cambio toglie g dal ciclo sulle celle.
     */
    backend->source_term =
        xmalloc_real_array(checked_mul((size_t)backend->batch_lines,
                                       (size_t)d->n[0], "batch g"),
                           "batch g");
    backend->abscissa =
        xmalloc_real_array((size_t)d->n[0], "line abscissas");

    /* La matrice della pressione non cambia mai: si scrive una volta qui. */
    for (int axis = 0; axis < 3; axis++) {
        backend->matrix_a[axis] =
            xmalloc_real_array((size_t)d->n[axis], "pressure matrix a");
        backend->matrix_b[axis] =
            xmalloc_real_array((size_t)d->n[axis], "pressure matrix b");
        backend->matrix_c[axis] =
            xmalloc_real_array((size_t)d->n[axis], "pressure matrix c");
        pressure_matrix(d, axis, backend->matrix_a[axis],
                        backend->matrix_b[axis], backend->matrix_c[axis]);
    }

    solver_mem_state->backend = backend;
}

void backend_free(SolverMemState *solver_mem_state) {
    PipelineBackend *backend = solver_mem_state->backend;

    if (backend == NULL) {
        return;
    }
    for (int axis = 0; axis < 3; axis++) {
        free(backend->matrix_c[axis]);
        free(backend->matrix_b[axis]);
        free(backend->matrix_a[axis]);
    }
    free(backend->abscissa);
    free(backend->source_term);
    free(backend->backward);
    free(backend->forward);
    free(backend->d_prime);
    free(backend->c_prime);
    free(backend);
    solver_mem_state->backend = NULL;
}

const char *backend_name(void) {
    return "pipeline";
}
