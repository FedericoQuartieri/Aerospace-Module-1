#include "momentum.h"

static Real *vector_component(VectorField *field, int component) {
    if (component == 0) return field->v_x;
    if (component == 1) return field->v_y;
    return field->v_z;
}

static const Real *const_vector_component(const VectorField *field,
                                          int component) {
    if (component == 0) return field->v_x;
    if (component == 1) return field->v_y;
    return field->v_z;
}

static size_t axis_line_count(const Domain *domain, int axis) {
    if (axis == AXIS_X) {
        return (size_t)domain->local[AXIS_Y] * domain->local[AXIS_Z];
    }
    if (axis == AXIS_Y) {
        return (size_t)domain->local[AXIS_X] * domain->local[AXIS_Z];
    }
    return (size_t)domain->local[AXIS_X] * domain->local[AXIS_Y];
}

static void line_coordinates(const Domain *domain, int axis, size_t line,
                             int level, int *i, int *j, int *k) {
    if (axis == AXIS_X) {
        *i = level;
        *j = (int)(line % (size_t)domain->local[AXIS_Y]);
        *k = (int)(line / (size_t)domain->local[AXIS_Y]);
    } else if (axis == AXIS_Y) {
        *i = (int)(line % (size_t)domain->local[AXIS_X]);
        *j = level;
        *k = (int)(line / (size_t)domain->local[AXIS_X]);
    } else {
        *i = (int)(line % (size_t)domain->local[AXIS_X]);
        *j = (int)(line / (size_t)domain->local[AXIS_X]);
        *k = level;
    }
}

static size_t scratch_index(const PipelineWorkspace *pipeline, int axis,
                            int component, size_t batch, int level,
                            int line, int length) {
    size_t batch_base = (size_t)component * pipeline->component_capacity +
                        batch * (size_t)length * pipeline->batch_lines;

    /* X keeps each physical row contiguous; Y and Z vectorize across lines. */
    if (axis == AXIS_X) {
        return batch_base + (size_t)line * length + (size_t)level;
    }
    return batch_base + (size_t)level * pipeline->batch_lines +
           (size_t)line;
}

static Real inverse_spacing_square(int axis) {
    if (axis == AXIS_X) return (Real)DX_INVERSE_SQUARE;
    if (axis == AXIS_Y) return (Real)DY_INVERSE_SQUARE;
    return (Real)DZ_INVERSE_SQUARE;
}

#if defined(USE_SIMD) && SIMD_AVAILABLE
static SimdReal momentum_weight_simd(SimdReal permeability, int axis) {
    SimdReal one = simd_set1((Real)1);
    SimdReal two = simd_set1((Real)2);
    SimdReal numerator = simd_set1((Real)(DT * NU));
    SimdReal beta = simd_add(
        one, simd_div(numerator, simd_mul(two, permeability)));
    SimdReal gamma = simd_div(numerator, simd_mul(two, beta));

    return simd_mul(gamma, simd_set1(-inverse_spacing_square(axis)));
}

static int forward_interior_simd(SolverMemState *state, int axis,
                                 int component, size_t batch,
                                 size_t first_line, int active,
                                 int level, int line, int length) {
    const Domain *domain = &state->domain;
    PipelineWorkspace *pipeline = &state->pipeline;
    size_t absolute_line = first_line + (size_t)line;
    int i, j, k;
    int global_axis = domain_global_index(domain, level, axis);

    if (global_axis == 0 || global_axis == domain->global[axis] - 1 ||
        line + SIMD_LANES > active) {
        return 0;
    }

    line_coordinates(domain, axis, absolute_line, level, &i, &j, &k);
    if (i + SIMD_LANES > domain->local[AXIS_X]) {
        return 0;
    }

    size_t field_index = domain_index(domain, i, j, k);
    size_t current = scratch_index(pipeline, axis, component, batch,
                                   level, line, length);
    const Real *previous_c;
    const Real *previous_d;
    const Real *permeability =
        const_vector_component(&state->k, component);
    const Real *source = axis == AXIS_Y
        ? const_vector_component(&state->eta, component)
        : const_vector_component(&state->zeta, component);
    const Real *target = axis == AXIS_Y
        ? const_vector_component(&state->zeta, component)
        : const_vector_component(&state->u, component);

    if (level == 0) {
        previous_c = &pipeline->forward[line];
        previous_d = &pipeline->forward[active + line];
    } else {
        size_t previous = scratch_index(pipeline, axis, component, batch,
                                        level - 1, line, length);
        previous_c = &pipeline->c_prime[previous];
        previous_d = &pipeline->d_prime[previous];
    }

    SimdReal one = simd_set1((Real)1);
    SimdReal two = simd_set1((Real)2);
    SimdReal w = momentum_weight_simd(simd_loadu(&permeability[field_index]),
                                      axis);
    SimdReal inverse_diagonal = simd_div(
        one, simd_sub(simd_sub(one, simd_mul(two, w)),
                      simd_mul(w, simd_loadu(previous_c))));
    SimdReal raw_rhs = simd_sub(simd_loadu(&source[field_index]),
                                simd_loadu(&target[field_index]));

    simd_storeu(&pipeline->c_prime[current],
                simd_mul(w, inverse_diagonal));
    simd_storeu(&pipeline->d_prime[current],
                simd_mul(simd_sub(raw_rhs,
                                  simd_mul(w, simd_loadu(previous_d))),
                         inverse_diagonal));
    return SIMD_LANES;
}
#endif

static Real momentum_raw_rhs(SolverMemState *state, const Data *data,
                             int axis, int component, int t_step,
                             int i, int j, int k, Real permeability,
                             size_t field_index) {
    Real *target;
    const Real *source;

    if (axis == AXIS_X) {
        target = vector_component(&state->eta, component);
        source = const_vector_component(&state->u, component);
        return source[field_index] +
               ((Real)DT / beta_from_k(permeability)) *
                   g_value(i, j, k, t_step, permeability,
                           state, data, component) -
               target[field_index];
    }
    if (axis == AXIS_Y) {
        target = vector_component(&state->zeta, component);
        source = const_vector_component(&state->eta, component);
    } else {
        target = vector_component(&state->u, component);
        source = const_vector_component(&state->zeta, component);
    }
    return source[field_index] - target[field_index];
}

static void eliminate_momentum_point(SolverMemState *state, Data *data,
                                     int axis, int component, int t_step,
                                     int i, int j, int k,
                                     Real previous_c, Real previous_d,
                                     Real *current_c, Real *current_d) {
    const Domain *domain = &state->domain;
    const Real *permeability =
        const_vector_component(&state->k, component);
    int global_i = domain_global_index(domain, i, AXIS_X);
    int global_j = domain_global_index(domain, j, AXIS_Y);
    int global_k = domain_global_index(domain, k, AXIS_Z);
    int global_axis = axis == AXIS_X ? global_i :
                      axis == AXIS_Y ? global_j : global_k;
    size_t index = domain_index(domain, i, j, k);

    if (global_axis == 0) {
        *current_c = (Real)0;
        *current_d = bc_left(data->bc_velocity, global_i, global_j, global_k,
                             t_step, component);
        return;
    }

    Real w = -gamma_from_k(permeability[index]) *
             inverse_spacing_square(axis);
    Real raw_rhs = momentum_raw_rhs(state, data, axis, component, t_step,
                                    i, j, k, permeability[index], index);

    if (global_axis == domain->global[axis] - 1) {
        Real right = bc_right(data->bc_velocity, global_i, global_j, global_k,
                              t_step, component);

        *current_c = (Real)0;
        if (component == axis) {
            *current_d = right;
        } else {
            Real inverse_diagonal = (Real)1 /
                (((Real)1 - (Real)3 * w) - w * previous_c);
            *current_d =
                (raw_rhs - (Real)2 * w * right - w * previous_d) *
                inverse_diagonal;
        }
        return;
    }

    Real inverse_diagonal = (Real)1 /
        (((Real)1 - (Real)2 * w) - w * previous_c);
    *current_c = w * inverse_diagonal;
    *current_d = (raw_rhs - w * previous_d) * inverse_diagonal;
}

static void forward_component(SolverMemState *state, Data *data,
                              int axis, int component, int t_step) {
    const Domain *domain = &state->domain;
    PipelineWorkspace *pipeline = &state->pipeline;
    const int batch_lines = pipeline->batch_lines;
    const int length = domain->local[axis];
    const size_t line_count = axis_line_count(domain, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 100 + 4 * axis + component;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t first_line = batch * (size_t)batch_lines;
        int active = (int)(line_count - first_line);

        if (active > batch_lines) active = batch_lines;
        if (domain->lower[axis] != MPI_PROC_NULL) {
            MPI_Recv(pipeline->forward, 2 * active, mpi_real_type(),
                     domain->lower[axis], tag, domain->cart,
                     MPI_STATUS_IGNORE);
        }

        if (axis == AXIS_X) {
            for (int line = 0; line < active; line++) {
                Real previous_c = domain->lower[axis] == MPI_PROC_NULL
                    ? (Real)0 : pipeline->forward[line];
                Real previous_d = domain->lower[axis] == MPI_PROC_NULL
                    ? (Real)0 : pipeline->forward[active + line];

                for (int level = 0; level < length; level++) {
                    int i, j, k;
                    Real current_c, current_d;
                    size_t index = scratch_index(pipeline, axis, component,
                                                 batch, level, line, length);

                    line_coordinates(domain, axis, first_line + (size_t)line,
                                     level, &i, &j, &k);
                    eliminate_momentum_point(state, data, axis, component,
                                             t_step, i, j, k,
                                             previous_c, previous_d,
                                             &current_c, &current_d);
                    pipeline->c_prime[index] = current_c;
                    pipeline->d_prime[index] = current_d;
                    previous_c = current_c;
                    previous_d = current_d;
                }
            }
        } else {
            for (int level = 0; level < length; level++) {
                int line = 0;
                while (line < active) {
                    int i, j, k;
                    Real previous_c, previous_d;
                    Real current_c, current_d;

#if defined(USE_SIMD) && SIMD_AVAILABLE
                    int advanced = forward_interior_simd(
                        state, axis, component, batch, first_line,
                        active, level, line, length);
                    if (advanced > 0) {
                        line += advanced;
                        continue;
                    }
#endif
                    size_t index = scratch_index(pipeline, axis, component,
                                                 batch, level, line, length);

                    if (level == 0) {
                        previous_c = domain->lower[axis] == MPI_PROC_NULL
                            ? (Real)0 : pipeline->forward[line];
                        previous_d = domain->lower[axis] == MPI_PROC_NULL
                            ? (Real)0 : pipeline->forward[active + line];
                    } else {
                        size_t previous = scratch_index(
                            pipeline, axis, component, batch,
                            level - 1, line, length);
                        previous_c = pipeline->c_prime[previous];
                        previous_d = pipeline->d_prime[previous];
                    }

                    line_coordinates(domain, axis, first_line + (size_t)line,
                                     level, &i, &j, &k);
                    eliminate_momentum_point(state, data, axis, component,
                                             t_step, i, j, k,
                                             previous_c, previous_d,
                                             &current_c, &current_d);
                    pipeline->c_prime[index] = current_c;
                    pipeline->d_prime[index] = current_d;
                    line++;
                }
            }
        }

        if (domain->upper[axis] != MPI_PROC_NULL) {
            for (int line = 0; line < active; line++) {
                size_t last = scratch_index(pipeline, axis, component, batch,
                                            length - 1, line, length);
                pipeline->forward[line] = pipeline->c_prime[last];
                pipeline->forward[active + line] = pipeline->d_prime[last];
            }
            MPI_Send(pipeline->forward, 2 * active, mpi_real_type(),
                     domain->upper[axis], tag, domain->cart);
        }
    }
}

static void backward_component(SolverMemState *state, int axis,
                               int component) {
    const Domain *domain = &state->domain;
    PipelineWorkspace *pipeline = &state->pipeline;
    Real *target = axis == AXIS_X
        ? vector_component(&state->eta, component)
        : axis == AXIS_Y
            ? vector_component(&state->zeta, component)
            : vector_component(&state->u, component);
    const int batch_lines = pipeline->batch_lines;
    const int length = domain->local[axis];
    const size_t line_count = axis_line_count(domain, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 200 + 4 * axis + component;

    for (size_t batch_plus_one = batch_count; batch_plus_one > 0;
         batch_plus_one--) {
        size_t batch = batch_plus_one - 1;
        size_t first_line = batch * (size_t)batch_lines;
        int active = (int)(line_count - first_line);

        if (active > batch_lines) active = batch_lines;
        if (domain->upper[axis] != MPI_PROC_NULL) {
            MPI_Recv(pipeline->backward, active, mpi_real_type(),
                     domain->upper[axis], tag, domain->cart,
                     MPI_STATUS_IGNORE);
        }

        if (axis == AXIS_X) {
            for (int line = 0; line < active; line++) {
                Real next = domain->upper[axis] == MPI_PROC_NULL
                    ? (Real)0 : pipeline->backward[line];

                for (int level = length - 1; level >= 0; level--) {
                    int i, j, k;
                    size_t scratch = scratch_index(
                        pipeline, axis, component, batch,
                        level, line, length);
                    Real solution = pipeline->d_prime[scratch] -
                                    pipeline->c_prime[scratch] * next;

                    line_coordinates(domain, axis,
                                     first_line + (size_t)line,
                                     level, &i, &j, &k);
                    target[domain_index(domain, i, j, k)] += solution;
                    next = solution;
                }
                pipeline->backward[line] = next;
            }
        } else {
            if (domain->upper[axis] == MPI_PROC_NULL) {
                for (int line = 0; line < active; line++) {
                    pipeline->backward[line] = (Real)0;
                }
            }

            for (int level = length - 1; level >= 0; level--) {
                int line = 0;
                while (line < active) {
                    int i, j, k;
                    size_t scratch = scratch_index(
                        pipeline, axis, component, batch,
                        level, line, length);

                    line_coordinates(domain, axis,
                                     first_line + (size_t)line,
                                     level, &i, &j, &k);

#if defined(USE_SIMD) && SIMD_AVAILABLE
                    if (line + SIMD_LANES <= active &&
                        i + SIMD_LANES <= domain->local[AXIS_X]) {
                        SimdReal solution = simd_sub(
                            simd_loadu(&pipeline->d_prime[scratch]),
                            simd_mul(simd_loadu(&pipeline->c_prime[scratch]),
                                     simd_loadu(&pipeline->backward[line])));
                        size_t field_index = domain_index(domain, i, j, k);
                        simd_storeu(&target[field_index],
                                    simd_add(simd_loadu(&target[field_index]),
                                             solution));
                        simd_storeu(&pipeline->backward[line], solution);
                        line += SIMD_LANES;
                        continue;
                    }
#endif
                    Real solution = pipeline->d_prime[scratch] -
                                    pipeline->c_prime[scratch] *
                                        pipeline->backward[line];
                    target[domain_index(domain, i, j, k)] += solution;
                    pipeline->backward[line] = solution;
                    line++;
                }
            }
        }

        if (domain->lower[axis] != MPI_PROC_NULL) {
            MPI_Send(pipeline->backward, active, mpi_real_type(),
                     domain->lower[axis], tag, domain->cart);
        }
    }
}

static void solve_momentum_direction(SolverMemState *state, Data *data,
                                     int axis, int t_step) {
    for (int component = 0; component < 3; component++) {
        forward_component(state, data, axis, component, t_step);
    }
    for (int component = 2; component >= 0; component--) {
        backward_component(state, axis, component);
    }
}

static void exchange_momentum_source_halos(SolverMemState *state) {
    Domain *domain = &state->domain;

    for (int component = 0; component < 3; component++) {
        domain_exchange_halo(domain, vector_component(&state->eta, component),
                             AXIS_X);
        domain_exchange_halo(domain, vector_component(&state->zeta, component),
                             AXIS_Y);
        domain_exchange_halo(domain, vector_component(&state->u, component),
                             AXIS_Z);
    }
    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        domain_exchange_halo(domain, state->pressure_star.v, axis);
    }
}

void momentum_step(SolverMemState *state, Data *data, int t_step,
                   SolverStats *stats) {
    uint64_t start_ns;

    start_ns = time_ns();
    exchange_momentum_source_halos(state);
    stats->momentum_halo += time_ns() - start_ns;

    start_ns = time_ns();
    solve_momentum_direction(state, data, AXIS_X, t_step);
    stats->eta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    solve_momentum_direction(state, data, AXIS_Y, t_step);
    stats->zeta_sys += time_ns() - start_ns;

    start_ns = time_ns();
    solve_momentum_direction(state, data, AXIS_Z, t_step);
    stats->u_sys += time_ns() - start_ns;
}
