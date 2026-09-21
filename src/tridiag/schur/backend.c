#include "backend.h"
#include "backend_schur.h"
#include "momentum.h"
#include "pressure.h"
#include "utils.h"
#include "workers.h"

#include <stdlib.h>

/*
 * The scratch of the Schur complement.
 *
 * Before, it lived on the stack of solver_solve, which therefore had to know
 * both momentum_scratch_slice and SchurPlan: two details of this backend in a
 * shared file. Now it reaches it through SolverMemState.backend, and solver.c
 * no longer knows they exist.
 */

void backend_init(const Decomp *d, SolverMemState *solver_mem_state) {
    SchurBackend *backend = xmalloc(sizeof(SchurBackend));

    /*
     * The scalar solvers use one grid line; the SIMD kernels keep several
     * independent lines interleaved in the same buffers. One copy per thread,
     * one behind the other: the kernels take their own with
     * momentum_scratch_slice, which is the same formula used here.
     */
    backend->scratch_size =
        momentum_scratch_slice(d) * (size_t)workers_available();
    backend->rhs = xmalloc(backend->scratch_size * sizeof(Real));
    backend->tmp = xmalloc(backend->scratch_size * sizeof(Real));

    /* The three pressure matrices do not depend on the time step: they are
     * prepared now, once and for all. */
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

int backend_batch_lines(void) {
    return 0;
}
