#include "backend.h"
#include "backend_pipeline.h"
#include "parallel.h"
#include "pressure_common.h"
#include "utils.h"
#include "workers.h"

/* Coefficients were factored once, with the incoming MPI c' incorporated.
 * A tile owns complete lines; only the RHS is propagated at runtime. */
static void forward_tile(const Decomp *d, const PipelineAxis *p, size_t first,
                         int active, int line, const Real *source, Real *dp,
                         const Real *incoming) {
    int cell[3];
    pipeline_cell(d, p->axis, first + (size_t)line, 0, cell);
    size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
    int width = p->axis == 0 ? 1 : SIMD_LANES;
    if (width > active - line) width = active - line;
#if SIMD_AVAILABLE
    if (p->axis != 0 && width == SIMD_LANES &&
        cell[0] <= d->n[0] - SIMD_LANES) {
        SimdReal prev = incoming ? simd_loadu(incoming + line) : simd_set1(0);
        for (int level = 0; level < p->length; level++, here += d->stride[p->axis]) {
            prev = simd_mul(simd_sub(simd_loadu(source + here),
                                    simd_mul(simd_set1(p->pressure_a[level]), prev)),
                            simd_set1(p->pressure_inverse[level]));
            simd_storeu(dp + pipeline_at(p, level, line), prev);
        }
        return;
    }
#endif
    for (int lane = 0; lane < width; lane++) {
        int l = line + lane;
        pipeline_cell(d, p->axis, first + (size_t)l, 0, cell);
        here = decomp_index(d, cell[0], cell[1], cell[2]);
        Real prev = incoming ? incoming[l] : (Real)0;
        for (int level = 0; level < p->length; level++, here += d->stride[p->axis]) {
            prev = (source[here] - p->pressure_a[level] * prev) *
                   p->pressure_inverse[level];
            dp[pipeline_at(p, level, l)] = prev;
        }
    }
}

static void backward_tile(const Decomp *d, const PipelineAxis *p, size_t first,
                          int active, int line, const Real *dp, Real *target,
                          Real *junction) {
    int cell[3];
    pipeline_cell(d, p->axis, first + (size_t)line, p->length - 1, cell);
    size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
    int width = p->axis == 0 ? 1 : SIMD_LANES;
    if (width > active - line) width = active - line;
#if SIMD_AVAILABLE
    if (p->axis != 0 && width == SIMD_LANES &&
        cell[0] <= d->n[0] - SIMD_LANES) {
        SimdReal x = junction && p->has_upper ? simd_loadu(junction + line)
                                             : simd_set1(0);
        for (int level = p->length - 1; level >= 0; level--) {
            x = simd_sub(simd_loadu(dp + pipeline_at(p, level, line)),
                         simd_mul(simd_set1(p->pressure_c[level]), x));
            simd_storeu(target + here, x);
            if (level > 0) here -= d->stride[p->axis];
        }
        if (junction) simd_storeu(junction + line, x);
        return;
    }
#endif
    for (int lane = 0; lane < width; lane++) {
        int l = line + lane;
        pipeline_cell(d, p->axis, first + (size_t)l, p->length - 1, cell);
        here = decomp_index(d, cell[0], cell[1], cell[2]);
        Real x = junction && p->has_upper ? junction[l] : (Real)0;
        for (int level = p->length - 1; level >= 0; level--) {
            x = dp[pipeline_at(p, level, l)] - p->pressure_c[level] * x;
            target[here] = x;
            if (level > 0) here -= d->stride[p->axis];
        }
        if (junction) junction[l] = x;
    }
}

void pipeline_pressure_direction(const Decomp *d, PipelineBackend *backend,
                                 int axis, const Real *source, Real *target) {
    const PipelineAxis *p = &backend->axes[axis];
    int tile = axis == 0 ? 1 : SIMD_LANES;
    int distributed = pipeline_distributed(p);
    WORKERS_PARALLEL(workers_many() &&
                     (distributed ? p->batch_lines > tile : p->batches > 1))
    {
        if (!distributed) {
            Real *dp = backend->d_prime + (size_t)workers_id() * backend->local_capacity;
            WORKERS_FOR
            for (size_t batch = 0; batch < p->batches; batch++) {
                int active = pipeline_active(p, batch);
                size_t first = batch * (size_t)p->batch_lines;
                for (int line = 0; line < active; line += tile)
                    forward_tile(d, p, first, active, line, source, dp, NULL);
                for (int line = 0; line < active; line += tile)
                    backward_tile(d, p, first, active, line, dp, target, NULL);
            }
        } else {
            for (size_t batch = 0; batch < p->batches; batch++) {
                int active = pipeline_active(p, batch);
                size_t first = batch * (size_t)p->batch_lines;
                Real *dp = backend->d_prime + pipeline_batch_offset(p, batch);
                WORKERS_MASTER
                {
                    if (p->has_lower)
                        par_recv_real(axis, -1, backend->forward, active, 300 + axis);
                }
                WORKERS_BARRIER
                WORKERS_FOR
                for (int line = 0; line < active; line += tile)
                    forward_tile(d, p, first, active, line, source, dp,
                                 p->has_lower ? backend->forward : NULL);
                WORKERS_MASTER
                {
                    if (p->has_upper) {
                        for (int line = 0; line < active; line++)
                            backend->forward[line] = dp[pipeline_at(p, p->length - 1, line)];
                        par_send_real(axis, 1, backend->forward, active, 300 + axis);
                    }
                }
                WORKERS_BARRIER
            }
            for (size_t remaining = p->batches; remaining > 0; remaining--) {
                size_t batch = remaining - 1;
                int active = pipeline_active(p, batch);
                size_t first = batch * (size_t)p->batch_lines;
                WORKERS_MASTER
                {
                    if (p->has_upper)
                        par_recv_real(axis, 1, backend->backward, active, 400 + axis);
                }
                WORKERS_BARRIER
                WORKERS_FOR
                for (int line = 0; line < active; line += tile)
                    backward_tile(d, p, first, active, line,
                                  backend->d_prime + pipeline_batch_offset(p, batch),
                                  target, backend->backward);
                WORKERS_MASTER
                {
                    if (p->has_lower)
                        par_send_real(axis, -1, backend->backward, active, 400 + axis);
                }
                WORKERS_BARRIER
            }
        }
    }
}

void pressure_step(const Decomp *d, SolverMemState *state,
                   ScalarField *pressure_buffer, SolverStats *stats) {
    PipelineBackend *backend = state->backend;
    Real *buffer = pressure_buffer->v, *star = state->pressure_star.v;
    uint64_t start = time_ns();
    compute_div(d, buffer, &state->u);
    pipeline_pressure_direction(d, backend, 0, buffer, star);
    stats->psi_sys += time_ns() - start;
    start = time_ns();
    pipeline_pressure_direction(d, backend, 1, star, buffer);
    stats->phi_low_sys += time_ns() - start;
    start = time_ns();
    pipeline_pressure_direction(d, backend, 2, buffer, star);
    stats->phi_high_sys += time_ns() - start;
    start = time_ns();
    update_pressure(d, state);
    stats->pressure_update += time_ns() - start;
}
