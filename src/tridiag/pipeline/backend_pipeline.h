#ifndef BACKEND_PIPELINE_H
#define BACKEND_PIPELINE_H

#include <limits.h>
#include <stddef.h>
#include "decomp.h"
#include "solver.h"
#include "simd_real.h"

#if defined(PIPELINE_BATCH_LINES) && (PIPELINE_BATCH_LINES < 1 || PIPELINE_BATCH_LINES > INT_MAX / 2)
#error "PIPELINE_BATCH_LINES must be in [1, INT_MAX / 2]"
#endif

/* Geometry is cached before entering OpenMP: even MPI_Cart_shift must be
 * called by the initializing thread when MPI_THREAD_FUNNELED is in use. */
typedef struct PipelineAxis {
    int axis, length, batch_lines;
    int has_lower, has_upper;
    size_t lines, batches;
    Real *pressure_a;
    Real *pressure_c;       /* globally propagated normalized superdiagonal */
    Real *pressure_inverse;
} PipelineAxis;

typedef struct PipelineBackend {
    PipelineAxis axes[3];
    size_t component_capacity; /* full local volume, for distributed axes */
    size_t local_capacity;     /* one reusable batch per OpenMP thread */
    size_t scratch_count;      /* elements in EACH of c_prime and d_prime */
    Real *c_prime, *d_prime;
    Real *forward, *backward;
    Real *source_term, *abscissa; /* one g line per worker, three coordinate lines */
    uint64_t *source_ns;
    uint64_t last_source_ns;
    int worker_slots;
} PipelineBackend;

static inline size_t pipeline_line_count(const Decomp *d, int axis) {
    if (axis == 0) return (size_t)d->n[1] * (size_t)d->n[2];
    if (axis == 1) return (size_t)d->n[0] * (size_t)d->n[2];
    return (size_t)d->n[0] * (size_t)d->n[1];
}

static inline int pipeline_distributed(const PipelineAxis *p) {
    return p->has_lower || p->has_upper;
}

static inline int pipeline_active(const PipelineAxis *p, size_t batch) {
    size_t remaining = p->lines - batch * (size_t)p->batch_lines;
    return (int)(remaining < (size_t)p->batch_lines ? remaining
                                                  : (size_t)p->batch_lines);
}

static inline void pipeline_cell(const Decomp *d, int axis, size_t line,
                                 int level, int cell[3]) {
    if (axis == 0) {
        cell[0] = level;
        cell[1] = (int)(line % (size_t)d->n[1]);
        cell[2] = (int)(line / (size_t)d->n[1]);
    } else if (axis == 1) {
        cell[0] = (int)(line % (size_t)d->n[0]);
        cell[1] = level;
        cell[2] = (int)(line / (size_t)d->n[0]);
    } else {
        cell[0] = (int)(line % (size_t)d->n[0]);
        cell[1] = (int)(line / (size_t)d->n[0]);
        cell[2] = level;
    }
}

/* X streams a complete contiguous line; Y/Z put neighboring lines together
 * so a SIMD vector can traverse the entire recurrence without a barrier. */
static inline size_t pipeline_at(const PipelineAxis *p, int level, int line) {
    return p->axis == 0 ? (size_t)line * (size_t)p->length + (size_t)level
                       : (size_t)level * (size_t)p->batch_lines + (size_t)line;
}

static inline size_t pipeline_batch_offset(const PipelineAxis *p, size_t batch) {
    return batch * (size_t)p->length * (size_t)p->batch_lines;
}

/* Private backend entry points, also used by the residual tests. */
void pipeline_momentum_direction(const Decomp *d, SolverMemState *state,
                                 Data *data, int axis, int t_step);
void pipeline_pressure_direction(const Decomp *d, PipelineBackend *backend,
                                 int axis, const Real *source, Real *target);
#endif
