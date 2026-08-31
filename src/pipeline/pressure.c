#include "pressure.h"

static size_t pressure_line_count(const Domain *domain, int axis) {
    if (axis == AXIS_X) {
        return (size_t)domain->local[AXIS_Y] * domain->local[AXIS_Z];
    }
    if (axis == AXIS_Y) {
        return (size_t)domain->local[AXIS_X] * domain->local[AXIS_Z];
    }
    return (size_t)domain->local[AXIS_X] * domain->local[AXIS_Y];
}

static void pressure_line_coordinates(const Domain *domain, int axis,
                                      size_t line, int level,
                                      int *i, int *j, int *k) {
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

static size_t pressure_scratch_index(const PipelineWorkspace *pipeline,
                                     int axis, size_t batch, int level,
                                     int line, int length) {
    size_t batch_base = batch * (size_t)length * pipeline->batch_lines;

    if (axis == AXIS_X) {
        return batch_base + (size_t)line * length + (size_t)level;
    }
    return batch_base + (size_t)level * pipeline->batch_lines +
           (size_t)line;
}

static Real pressure_weight(int axis) {
    if (axis == AXIS_X) return -(Real)DX_INVERSE_SQUARE;
    if (axis == AXIS_Y) return -(Real)DY_INVERSE_SQUARE;
    return -(Real)DZ_INVERSE_SQUARE;
}

static Real divergence_rhs(const SolverMemState *state,
                           int i, int j, int k) {
    const Domain *domain = &state->domain;
    int global_i = domain_global_index(domain, i, AXIS_X);
    int global_j = domain_global_index(domain, j, AXIS_Y);
    int global_k = domain_global_index(domain, k, AXIS_Z);
    size_t index = domain_index(domain, i, j, k);

    if (global_i == 0 || global_j == 0 || global_k == 0) {
        return (Real)0;
    }

    return -((state->u.v_x[index] - state->u.v_x[index - 1]) *
                 (Real)DX_INVERSE +
             (state->u.v_y[index] -
              state->u.v_y[index - domain->stride_y]) *
                 (Real)DY_INVERSE +
             (state->u.v_z[index] -
              state->u.v_z[index - domain->stride_z]) *
                 (Real)DZ_INVERSE) /
           (Real)DT;
}

static Real pressure_raw_rhs(const SolverMemState *state,
                             const ScalarField *input, int axis,
                             int i, int j, int k) {
    if (axis == AXIS_X) {
        return divergence_rhs(state, i, j, k);
    }
    return input->v[domain_index(&state->domain, i, j, k)];
}

static void eliminate_pressure_point(const SolverMemState *state,
                                     const ScalarField *input, int axis,
                                     int i, int j, int k,
                                     Real previous_c, Real previous_d,
                                     Real *current_c, Real *current_d) {
    const Domain *domain = &state->domain;
    int global_axis = domain_global_index(domain,
                                          axis == AXIS_X ? i :
                                          axis == AXIS_Y ? j : k,
                                          axis);
    Real w = pressure_weight(axis);
    Real raw_rhs = pressure_raw_rhs(state, input, axis, i, j, k);
    Real inverse_diagonal;

    if (global_axis == 0) {
        inverse_diagonal = (Real)1 / ((Real)1 - (Real)2 * w);
        *current_c = (Real)2 * w * inverse_diagonal;
        *current_d = raw_rhs * inverse_diagonal;
    } else if (global_axis == domain->global[axis] - 1) {
        inverse_diagonal =
            (Real)1 / (((Real)1 - w) - w * previous_c);
        *current_c = (Real)0;
        *current_d = (raw_rhs - w * previous_d) * inverse_diagonal;
    } else {
        inverse_diagonal =
            (Real)1 / (((Real)1 - (Real)2 * w) - w * previous_c);
        *current_c = w * inverse_diagonal;
        *current_d = (raw_rhs - w * previous_d) * inverse_diagonal;
    }
}

static void pressure_forward(SolverMemState *state,
                             const ScalarField *input, int axis) {
    const Domain *domain = &state->domain;
    PipelineWorkspace *pipeline = &state->pipeline;
    const int batch_lines = pipeline->batch_lines;
    const int length = domain->local[axis];
    const size_t line_count = pressure_line_count(domain, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 300 + axis;

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
                    size_t scratch = pressure_scratch_index(
                        pipeline, axis, batch, level, line, length);

                    pressure_line_coordinates(domain, axis,
                                              first_line + (size_t)line,
                                              level, &i, &j, &k);
                    eliminate_pressure_point(state, input, axis, i, j, k,
                                             previous_c, previous_d,
                                             &current_c, &current_d);
                    pipeline->c_prime[scratch] = current_c;
                    pipeline->d_prime[scratch] = current_d;
                    previous_c = current_c;
                    previous_d = current_d;
                }
            }
        } else {
            for (int level = 0; level < length; level++) {
                for (int line = 0; line < active; line++) {
                    int i, j, k;
                    Real previous_c, previous_d;
                    Real current_c, current_d;
                    size_t scratch = pressure_scratch_index(
                        pipeline, axis, batch, level, line, length);

                    if (level == 0) {
                        previous_c = domain->lower[axis] == MPI_PROC_NULL
                            ? (Real)0 : pipeline->forward[line];
                        previous_d = domain->lower[axis] == MPI_PROC_NULL
                            ? (Real)0 : pipeline->forward[active + line];
                    } else {
                        size_t previous = pressure_scratch_index(
                            pipeline, axis, batch, level - 1, line, length);
                        previous_c = pipeline->c_prime[previous];
                        previous_d = pipeline->d_prime[previous];
                    }

                    pressure_line_coordinates(domain, axis,
                                              first_line + (size_t)line,
                                              level, &i, &j, &k);
                    eliminate_pressure_point(state, input, axis, i, j, k,
                                             previous_c, previous_d,
                                             &current_c, &current_d);
                    pipeline->c_prime[scratch] = current_c;
                    pipeline->d_prime[scratch] = current_d;
                }
            }
        }

        if (domain->upper[axis] != MPI_PROC_NULL) {
            for (int line = 0; line < active; line++) {
                size_t last = pressure_scratch_index(
                    pipeline, axis, batch, length - 1, line, length);
                pipeline->forward[line] = pipeline->c_prime[last];
                pipeline->forward[active + line] = pipeline->d_prime[last];
            }
            MPI_Send(pipeline->forward, 2 * active, mpi_real_type(),
                     domain->upper[axis], tag, domain->cart);
        }
    }
}

static void pressure_backward(SolverMemState *state,
                              ScalarField *output, int axis) {
    const Domain *domain = &state->domain;
    PipelineWorkspace *pipeline = &state->pipeline;
    const int batch_lines = pipeline->batch_lines;
    const int length = domain->local[axis];
    const size_t line_count = pressure_line_count(domain, axis);
    const size_t batch_count =
        (line_count + (size_t)batch_lines - 1) / (size_t)batch_lines;
    const int tag = 320 + axis;

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
                    size_t scratch = pressure_scratch_index(
                        pipeline, axis, batch, level, line, length);
                    Real solution = pipeline->d_prime[scratch] -
                                    pipeline->c_prime[scratch] * next;

                    pressure_line_coordinates(domain, axis,
                                              first_line + (size_t)line,
                                              level, &i, &j, &k);
                    output->v[domain_index(domain, i, j, k)] = solution;
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
                for (int line = 0; line < active; line++) {
                    int i, j, k;
                    size_t scratch = pressure_scratch_index(
                        pipeline, axis, batch, level, line, length);
                    Real solution = pipeline->d_prime[scratch] -
                                    pipeline->c_prime[scratch] *
                                        pipeline->backward[line];

                    pressure_line_coordinates(domain, axis,
                                              first_line + (size_t)line,
                                              level, &i, &j, &k);
                    output->v[domain_index(domain, i, j, k)] = solution;
                    pipeline->backward[line] = solution;
                }
            }
        }

        if (domain->lower[axis] != MPI_PROC_NULL) {
            MPI_Send(pipeline->backward, active, mpi_real_type(),
                     domain->lower[axis], tag, domain->cart);
        }
    }
}

static void solve_pressure_direction(SolverMemState *state,
                                     const ScalarField *input,
                                     ScalarField *output, int axis) {
    pressure_forward(state, input, axis);
    pressure_backward(state, output, axis);
}

static void exchange_velocity_for_divergence(SolverMemState *state) {
    domain_exchange_halo(&state->domain, state->u.v_x, AXIS_X);
    domain_exchange_halo(&state->domain, state->u.v_y, AXIS_Y);
    domain_exchange_halo(&state->domain, state->u.v_z, AXIS_Z);
}

static void update_pressure(SolverMemState *state) {
    const Domain *domain = &state->domain;

    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            for (int i = 0; i < domain->local[AXIS_X]; i++) {
                size_t index = domain_index(domain, i, j, k);
                Real phi = state->pressure_star.v[index];
                Real pressure_new = state->pressure.v[index] + phi;

                state->pressure.v[index] = pressure_new;
                state->pressure_star.v[index] = pressure_new + phi;
            }
        }
    }
}

void pressure_step(SolverMemState *state, ScalarField *pressure_buffer,
                   SolverStats *stats) {
    uint64_t start_ns;

    start_ns = time_ns();
    exchange_velocity_for_divergence(state);
    stats->pressure_halo += time_ns() - start_ns;

    start_ns = time_ns();
    solve_pressure_direction(state, NULL, &state->pressure_star, AXIS_X);
    stats->psi_sys += time_ns() - start_ns;

    start_ns = time_ns();
    solve_pressure_direction(state, &state->pressure_star,
                             pressure_buffer, AXIS_Y);
    stats->phi_low_sys += time_ns() - start_ns;

    start_ns = time_ns();
    solve_pressure_direction(state, pressure_buffer,
                             &state->pressure_star, AXIS_Z);
    stats->phi_high_sys += time_ns() - start_ns;

    start_ns = time_ns();
    update_pressure(state);
    stats->pressure_update += time_ns() - start_ns;
}
