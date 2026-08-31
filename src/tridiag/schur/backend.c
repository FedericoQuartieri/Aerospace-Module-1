#include "backend.h"
#include "backend_schur.h"
#include "momentum.h"
#include "pressure.h"
#include "utils.h"
#include "workers.h"

#include <stdlib.h>

/*
 * Lo scratch del complemento di Schur.
 *
 * Prima stava sullo stack di solver_solve, che percio' doveva conoscere sia
 * momentum_scratch_slice sia SchurPlan: due dettagli di questo backend in un
 * file condiviso.  Ora ci arriva attraverso SolverMemState.backend, e
 * solver.c non sa piu' che esistano.
 */

void backend_init(const Decomp *d, SolverMemState *solver_mem_state) {
    SchurBackend *backend = xmalloc(sizeof(SchurBackend));

    /*
     * I solutori scalari usano una linea di griglia; i kernel SIMD tengono
     * piu' linee indipendenti interlacciate negli stessi buffer.  Una copia
     * per thread, una dietro l'altra: i kernel si prendono la propria con
     * momentum_scratch_slice, che e' la stessa formula usata qui.
     */
    backend->scratch_size =
        momentum_scratch_slice(d) * (size_t)workers_available();
    backend->rhs = xmalloc(backend->scratch_size * sizeof(Real));
    backend->tmp = xmalloc(backend->scratch_size * sizeof(Real));

    /* Le tre matrici della pressione non dipendono dal passo temporale: si
     * preparano adesso, una volta per tutte. */
    pressure_plans_init(d, backend->pressure_plan);

    solver_mem_state->backend = backend;
}

void backend_free(SolverMemState *solver_mem_state) {
    SchurBackend *backend = solver_mem_state->backend;

    if (backend == NULL) {
        return;
    }
    pressure_plans_free(backend->pressure_plan);
    free(backend->tmp);
    free(backend->rhs);
    free(backend);
    solver_mem_state->backend = NULL;
}

const char *backend_name(void) {
    return "schur";
}
