#include "backend.h"
#include "backend_pipeline.h"
#include "momentum_row.h"
#include "parallel.h"
#include "utils.h"
#include "workers.h"

static Real *target_field(SolverMemState *s, int axis, int component) {
    VectorField *v = axis == 0 ? &s->eta : axis == 1 ? &s->zeta : &s->u;
    return component == 0 ? v->v_x : component == 1 ? v->v_y : v->v_z;
}

#if SIMD_AVAILABLE
static SimdReal momentum_weight_simd(SimdReal permeability,
                                     Real inverse_square) {
    SimdReal one = simd_set1((Real)1);
    SimdReal two = simd_set1((Real)2);
    SimdReal numerator = simd_set1((Real)(DT * NU));
    SimdReal beta = simd_add(
        one, simd_div(numerator, simd_mul(two, permeability)));
    SimdReal gamma = simd_div(numerator, simd_mul(two, beta));

    return simd_mul(gamma, simd_set1(-inverse_square));
}

#endif

/* A tile owns complete independent lines. No thread synchronization is
 * needed between levels. X uses a contiguous scalar line; Y/Z use SIMD
 * across adjacent lines, including on distributed axes. */
static void forward_tile(const MomentumLine *ctx, const PipelineAxis *p,
                         size_t first, int active, int line, Real *cp, Real *dp,
                         const Real *incoming, Real *source_term,
                         const Real *abscissa, uint64_t *source_ns) {
    const Decomp *d = ctx->d;
    int cell[3];
    pipeline_cell(d, p->axis, first + (size_t)line, 0, cell);
    size_t here = decomp_index(d, cell[0], cell[1], cell[2]);
    int width = p->axis == 0 ? 1 : SIMD_LANES;
    if (width > active - line) width = active - line;
#if SIMD_AVAILABLE
    if (p->axis != 0 && width == SIMD_LANES &&
        cell[0] <= d->n[0] - SIMD_LANES) {
        SimdReal pc = incoming ? simd_loadu(incoming + line) : simd_set1(0);
        SimdReal pd = incoming ? simd_loadu(incoming + active + line) : simd_set1(0);
        for (int level = 0; level < p->length; level++, here += d->stride[p->axis]) {
            int global = d->start[p->axis] + level;
            size_t at = pipeline_at(p, level, line);
            if (global > 0 && global < ctx->last_global) {
                SimdReal w = momentum_weight_simd(simd_loadu(ctx->k_porosity + here),
                                                  ctx->inverse_square);
                SimdReal b = simd_sub(simd_set1(1), simd_mul(simd_set1(2), w));
                SimdReal inv = simd_div(simd_set1(1), simd_sub(b, simd_mul(w, pc)));
                SimdReal rhs = simd_sub(simd_loadu(ctx->source + here),
                                        simd_loadu(ctx->target + here));
                pc = simd_mul(w, inv);
                pd = simd_mul(simd_sub(rhs, simd_mul(w, pd)), inv);
                simd_storeu(cp + at, pc);
                simd_storeu(dp + at, pd);
            } else {
                Real previous_c[SIMD_LANES], previous_d[SIMD_LANES];
                simd_storeu(previous_c, pc);
                simd_storeu(previous_d, pd);
                cell[p->axis] = level;
                for (int lane = 0; lane < SIMD_LANES; lane++) {
                    cell[0] += lane;
                    MomentumRow row = momentum_row(ctx, cell, here + (size_t)lane);
                    cell[0] -= lane;
                    Real inv = (Real)1 / (row.b - row.a * previous_c[lane]);
                    cp[at + (size_t)lane] = row.c * inv;
                    dp[at + (size_t)lane] = (row.f - row.a * previous_d[lane]) * inv;
                }
                pc = simd_loadu(cp + at);
                pd = simd_loadu(dp + at);
            }
        }
        return;
    }
#endif
    MomentumLine prepared = *ctx;
    if (p->axis == 0) {
        uint64_t start = time_ns();
        g_line(d, ctx->data, ctx->state, ctx->k_porosity, cell[1], cell[2],
               ctx->t_step, ctx->v_comp, abscissa, source_term);
        *source_ns += time_ns() - start;
        prepared.source_term = source_term;
    }
    for (int lane = 0; lane < width; lane++) {
        int l = line + lane;
        pipeline_cell(d, p->axis, first + (size_t)l, 0, cell);
        here = decomp_index(d, cell[0], cell[1], cell[2]);
        Real pc = incoming ? incoming[l] : (Real)0;
        Real pd = incoming ? incoming[active + l] : (Real)0;
        for (int level = 0; level < p->length; level++, here += d->stride[p->axis]) {
            cell[p->axis] = level;
            MomentumRow row = momentum_row(&prepared, cell, here);
            Real inv = (Real)1 / (row.b - row.a * pc);
            size_t at = pipeline_at(p, level, l);
            pc = cp[at] = row.c * inv;
            pd = dp[at] = (row.f - row.a * pd) * inv;
        }
    }
}

static void backward_tile(const Decomp *d, const PipelineAxis *p,
                          size_t first, int active, int line,
                          const Real *cp, const Real *dp, Real *target,
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
            size_t at = pipeline_at(p, level, line);
            x = simd_sub(simd_loadu(dp + at), simd_mul(simd_loadu(cp + at), x));
            simd_storeu(target + here, simd_add(simd_loadu(target + here), x));
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
            size_t at = pipeline_at(p, level, l);
            x = dp[at] - cp[at] * x;
            target[here] += x;
            if (level > 0) here -= d->stride[p->axis];
        }
        if (junction) junction[l] = x;
    }
}

void pipeline_momentum_direction(const Decomp *d, SolverMemState *state,
                                 Data *data, int axis, int t_step) {
    PipelineBackend *backend = state->backend;
    const PipelineAxis *p = &backend->axes[axis];
    int tile = axis == 0 ? 1 : SIMD_LANES;
    int distributed = pipeline_distributed(p);

    for (int w = 0; w < backend->worker_slots; w++) backend->source_ns[w] = 0;

    WORKERS_PARALLEL(workers_many() &&
                     (distributed ? p->batch_lines > tile : p->batches > 1))
    {
        uint64_t source_ns = 0;
        Real *source_term = backend->source_term + (size_t)workers_id() * (size_t)d->n[0];
        if (!distributed) {
            Real *cp = backend->c_prime + (size_t)workers_id() * backend->local_capacity;
            Real *dp = backend->d_prime + (size_t)workers_id() * backend->local_capacity;
            for (int component = 0; component < 3; component++) {
                MomentumLine ctx = momentum_line(d, state, data, t_step, component, axis);
                Real *target = target_field(state, axis, component);
                WORKERS_FOR
                for (size_t batch = 0; batch < p->batches; batch++) {
                    int active = pipeline_active(p, batch);
                    size_t first = batch * (size_t)p->batch_lines;
                    /* For X, g reads the old eta on the SAME line only.
                     * Complete its RHS before modifying that line. Y/Z read
                     * source and target pointwise, so batches are independent. */
                    for (int line = 0; line < active; line += tile)
                        forward_tile(&ctx, p, first, active, line, cp, dp, NULL, source_term,
                                     backend->abscissa + (size_t)component * (size_t)d->n[0],
                                     &source_ns);
                    for (int line = 0; line < active; line += tile)
                        backward_tile(d, p, first, active, line, cp, dp, target, NULL);
                }
            }
        } else {
            /* All MPI calls, including topology queries in the wrappers,
             * remain on the initializing thread. Worksharing barriers delimit
             * batch ownership; there are no barriers along a line. */
            for (int component = 0; component < 3; component++) {
                MomentumLine ctx = momentum_line(d, state, data, t_step, component, axis);
                for (size_t batch = 0; batch < p->batches; batch++) {
                    int active = pipeline_active(p, batch);
                    size_t first = batch * (size_t)p->batch_lines;
                    size_t offset = (size_t)component * backend->component_capacity +
                                    pipeline_batch_offset(p, batch);
                    Real *cp = backend->c_prime + offset, *dp = backend->d_prime + offset;
                    WORKERS_MASTER
                    {
                        if (p->has_lower)
                            par_recv_real(axis, -1, backend->forward, 2 * active,
                                          100 + 4 * axis + component);
                    }
                    WORKERS_BARRIER
                    WORKERS_FOR
                    for (int line = 0; line < active; line += tile)
                        forward_tile(&ctx, p, first, active, line, cp, dp,
                                     p->has_lower ? backend->forward : NULL, source_term,
                                     backend->abscissa + (size_t)component * (size_t)d->n[0],
                                     &source_ns);
                    WORKERS_MASTER
                    {
                        if (p->has_upper) {
                            for (int line = 0; line < active; line++) {
                                size_t at = pipeline_at(p, p->length - 1, line);
                                backend->forward[line] = cp[at];
                                backend->forward[active + line] = dp[at];
                            }
                            par_send_real(axis, 1, backend->forward, 2 * active,
                                          100 + 4 * axis + component);
                        }
                    }
                    WORKERS_BARRIER
                }
            }
            for (int component = 2; component >= 0; component--) {
                Real *target = target_field(state, axis, component);
                for (size_t remaining = p->batches; remaining > 0; remaining--) {
                    size_t batch = remaining - 1;
                    int active = pipeline_active(p, batch);
                    size_t first = batch * (size_t)p->batch_lines;
                    size_t offset = (size_t)component * backend->component_capacity +
                                    pipeline_batch_offset(p, batch);
                    WORKERS_MASTER
                    {
                        if (p->has_upper)
                            par_recv_real(axis, 1, backend->backward, active,
                                          200 + 4 * axis + component);
                    }
                    WORKERS_BARRIER
                    WORKERS_FOR
                    for (int line = 0; line < active; line += tile)
                        backward_tile(d, p, first, active, line,
                                      backend->c_prime + offset, backend->d_prime + offset,
                                      target, backend->backward);
                    WORKERS_MASTER
                    {
                        if (p->has_lower)
                            par_send_real(axis, -1, backend->backward, active,
                                          200 + 4 * axis + component);
                    }
                    WORKERS_BARRIER
                }
            }
        }
        backend->source_ns[workers_id()] = source_ns;
    }
    backend->last_source_ns = 0;
    for (int w = 0; w < backend->worker_slots; w++)
        if (backend->source_ns[w] > backend->last_source_ns)
            backend->last_source_ns = backend->source_ns[w];
}

void momentum_step(const Decomp *d, SolverMemState *state, Data *data,
                   int t_step, SolverStats *stats) {
    uint64_t *timings[3] = {&stats->eta_sys, &stats->zeta_sys, &stats->u_sys};
    for (int axis = 0; axis < 3; axis++) {
        uint64_t start = time_ns();
        pipeline_momentum_direction(d, state, data, axis, t_step);
        *timings[axis] += time_ns() - start;
        if (axis == 0) stats->momentum_source += ((PipelineBackend *)state->backend)->last_source_ns;
    }
}
