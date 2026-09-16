#include "backend.h"
#include "backend_pipeline.h"
#include "parallel.h"
#include "params.h"
#include "pressure_common.h"
#include "utils.h"
#include "workers.h"

#include <stdint.h>
#include <stdlib.h>

static size_t checked_mul(size_t a, size_t b) {
    if (a != 0 && b > SIZE_MAX / a) {
        fprintf(stderr, "pipeline: scratch size overflow\n");
        par_abort(1);
    }
    return a * b;
}

static Real *real_array(size_t count) {
    return xmalloc(checked_mul(count, sizeof(Real)));
}

#if !defined(PIPELINE_BATCH_LINES)
/*
 * Linee per batch dai thread del processo.
 *
 * Mantiene la regola introdotta dalla campagna precedente. Il nuovo ciclo
 * assegna linee complete ai thread, quindi non paga piu' barriere per livello:
 * l'ottimo va rimisurato sul cluster prima di cambiare questa euristica.
 * Gli override a compilazione e da file rimangono disponibili.
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

void backend_init(const Decomp *d, SolverMemState *state) {
    PipelineBackend *backend = xmalloc(sizeof *backend);
    *backend = (PipelineBackend){0};
    int max_batch = 1;
    long long requested = par_max_long(requested_batch_lines());
    if (requested < 1 || requested > INT_MAX / 2) {
        fprintf(stderr, "pipeline: batch must be in [1, INT_MAX / 2]\n");
        par_abort(1);
    }
    chosen_batch_lines = (int)requested;
    backend->worker_slots = workers_available();

    for (int axis = 0; axis < 3; axis++) {
        PipelineAxis *p = &backend->axes[axis];
        p->axis = axis;
        p->length = d->n[axis];
        p->lines = pipeline_line_count(d, axis);
        if (p->length < 1 || p->lines == 0) {
            fprintf(stderr, "pipeline: empty local block\n");
            par_abort(1);
        }
        p->batch_lines = (int)(p->lines < (size_t)chosen_batch_lines
                                  ? p->lines : (size_t)chosen_batch_lines);
        p->has_lower = par_neighbor(axis, -1) != PAR_NO_NEIGHBOR;
        p->has_upper = par_neighbor(axis, 1) != PAR_NO_NEIGHBOR;
        /* Local lines have no MPI batching constraint. Keep at least one
         * batch per worker so a large requested batch cannot serialize them. */
        if (!pipeline_distributed(p)) {
            size_t share = (p->lines - 1) / (size_t)backend->worker_slots + 1;
            if (share < (size_t)p->batch_lines) p->batch_lines = (int)share;
        }
        p->batches = (p->lines - 1) / (size_t)p->batch_lines + 1;
        if (p->batch_lines > max_batch) max_batch = p->batch_lines;
        size_t batch_room = checked_mul((size_t)p->length,
                                        (size_t)p->batch_lines);
        if (pipeline_distributed(p)) {
            size_t capacity = checked_mul(p->batches, batch_room);
            if (capacity > backend->component_capacity)
                backend->component_capacity = capacity;
        } else if (batch_room > backend->local_capacity) {
            backend->local_capacity = batch_room;
        }

        /* The pressure matrix is identical for all lines and all time steps.
         * Propagate its factorization ONCE through the ranks, including the
         * incoming c' at an MPI boundary. Runtime messages need only d'. */
        p->pressure_a = real_array((size_t)p->length);
        p->pressure_inverse = real_array((size_t)p->length);
        p->pressure_c = real_array((size_t)p->length);
        pressure_matrix(d, axis, p->pressure_a, p->pressure_inverse,
                        p->pressure_c);
        Real previous = 0;
        if (p->has_lower) par_recv_real(axis, -1, &previous, 1, 500 + axis);
        for (int level = 0; level < p->length; level++) {
            Real inverse = (Real)1 / (p->pressure_inverse[level] -
                                      p->pressure_a[level] * previous);
            p->pressure_inverse[level] = inverse;
            p->pressure_c[level] *= inverse;
            previous = p->pressure_c[level];
        }
        if (p->has_upper) par_send_real(axis, 1, &previous, 1, 500 + axis);
    }

    /* Distributed momentum retains three components: 6*N*sizeof(Real)
     * bytes, 96 MiB for a 128^3 local double block without batch padding.
     * Local axes reuse one batch per thread, shared across components. */
    size_t distributed = checked_mul(3, backend->component_capacity);
    size_t local = checked_mul((size_t)workers_available(),
                               backend->local_capacity);
    backend->scratch_count = distributed > local ? distributed : local;
    backend->c_prime = real_array(backend->scratch_count);
    backend->d_prime = real_array(backend->scratch_count);
    backend->forward = real_array(checked_mul(2, (size_t)max_batch));
    backend->backward = real_array((size_t)max_batch);
    backend->source_term = real_array(checked_mul((size_t)backend->worker_slots,
                                                  (size_t)d->n[0]));
    backend->abscissa = real_array(checked_mul(3, (size_t)d->n[0]));
    backend->source_ns = xmalloc(checked_mul((size_t)backend->worker_slots,
                                            sizeof *backend->source_ns));
    for (int c = 0; c < 3; c++)
        forcing_line_coords(d, c, backend->abscissa + (size_t)c * (size_t)d->n[0]);
    state->backend = backend;
}

void backend_free(SolverMemState *state) {
    PipelineBackend *backend = state->backend;
    if (!backend) return;
    for (int axis = 0; axis < 3; axis++) {
        free(backend->axes[axis].pressure_a);
        free(backend->axes[axis].pressure_c);
        free(backend->axes[axis].pressure_inverse);
    }
    free(backend->source_ns);
    free(backend->abscissa);
    free(backend->source_term);
    free(backend->backward);
    free(backend->forward);
    free(backend->d_prime);
    free(backend->c_prime);
    free(backend);
    state->backend = NULL;
}

const char *backend_name(void) { return "pipeline"; }
