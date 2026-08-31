#include "backend.h"
#include "backend_pipeline.h"
#include "pressure_common.h"
#include "utils.h"

#include <stdlib.h>

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
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;

    return batch_count * (size_t)batch_lines * (size_t)d->n[axis];
}

void backend_init(const Decomp *d, SolverMemState *solver_mem_state) {
    PipelineBackend *backend = xmalloc(sizeof(PipelineBackend));
    size_t capacity = 0;

    backend->batch_lines = PIPELINE_BATCH_LINES;
    if (backend->batch_lines < 1) {
        backend->batch_lines = 1;
    }

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
    backend->c_prime = xmalloc(3 * capacity * sizeof(Real));
    backend->d_prime = xmalloc(3 * capacity * sizeof(Real));
    backend->forward =
        xmalloc(2 * (size_t)backend->batch_lines * sizeof(Real));
    backend->backward = xmalloc((size_t)backend->batch_lines * sizeof(Real));

    /* La matrice della pressione non cambia mai: si scrive una volta qui. */
    for (int axis = 0; axis < 3; axis++) {
        size_t room = (size_t)d->n[axis] * sizeof(Real);

        backend->matrix_a[axis] = xmalloc(room);
        backend->matrix_b[axis] = xmalloc(room);
        backend->matrix_c[axis] = xmalloc(room);
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
