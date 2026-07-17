/* DESIGN: the optimized backend keeps the scalar X recurrence and exposes
 * independent X-adjacent Y/Z systems to explicit SIMD.  Tail lines use the
 * same Thomas rows and the same preallocated scratch. */
#include "kernels.h"
#include "solver_internal.h"

#include <assert.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define OPTIMIZED_HAS_SIMD 1
#if defined(USE_FLOAT)
typedef float32x4_t SimdReal;
#define SIMD_LENGTH 4
#define SIMD_LOAD(pointer) vld1q_f32(pointer)
#define SIMD_STORE(pointer, value) vst1q_f32((pointer), (value))
#define SIMD_SET1(value) vdupq_n_f32(value)
#define SIMD_ADD(a, b) vaddq_f32((a), (b))
#define SIMD_SUB(a, b) vsubq_f32((a), (b))
#define SIMD_MUL(a, b) vmulq_f32((a), (b))
#define SIMD_DIV(a, b) vdivq_f32((a), (b))
#else
typedef float64x2_t SimdReal;
#define SIMD_LENGTH 2
#define SIMD_LOAD(pointer) vld1q_f64(pointer)
#define SIMD_STORE(pointer, value) vst1q_f64((pointer), (value))
#define SIMD_SET1(value) vdupq_n_f64(value)
#define SIMD_ADD(a, b) vaddq_f64((a), (b))
#define SIMD_SUB(a, b) vsubq_f64((a), (b))
#define SIMD_MUL(a, b) vmulq_f64((a), (b))
#define SIMD_DIV(a, b) vdivq_f64((a), (b))
#endif
#else
#define OPTIMIZED_HAS_SIMD 0
typedef Real SimdReal;
#define SIMD_LENGTH 1
#define SIMD_LOAD(pointer) (*(pointer))
#define SIMD_STORE(pointer, value) (*(pointer) = (value))
#define SIMD_SET1(value) (value)
#define SIMD_ADD(a, b) ((a) + (b))
#define SIMD_SUB(a, b) ((a) - (b))
#define SIMD_MUL(a, b) ((a) * (b))
#define SIMD_DIV(a, b) ((a) / (b))
#endif

static size_t maximum_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

size_t optimized_scratch_capacity(const Grid *grid)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t longest = maximum_size(grid->extent[DIRECTION_Y],
                                        grid->extent[DIRECTION_Z]);
    const size_t standard_need = maximum_size(3 * nx, 2 * nx * longest);
    const size_t largest_slice = 8 * SIMD_LENGTH;
    const size_t simd_need = 2 * longest * largest_slice + largest_slice +
                             3 * longest;
    return maximum_size(standard_need, simd_need);
}

static void pressure_thomas_line(Real w,
                                 size_t length,
                                 Real *tmp,
                                 Real *rhs,
                                 Real *solution)
{
    size_t index;
    Real inverse_diagonal = (Real)1 / ((Real)1 - (Real)2 * w);
    tmp[0] = ((Real)2 * w) * inverse_diagonal;
    rhs[0] *= inverse_diagonal;
    for (index = 1; index + 1 < length; ++index) {
        inverse_diagonal =
            (Real)1 /
            (((Real)1 - (Real)2 * w) - w * tmp[index - 1]);
        tmp[index] = w * inverse_diagonal;
        rhs[index] = (rhs[index] - w * rhs[index - 1]) *
                     inverse_diagonal;
    }
    inverse_diagonal =
        (Real)1 / (((Real)1 - w) - w * tmp[length - 2]);
    rhs[length - 1] =
        (rhs[length - 1] - w * rhs[length - 2]) * inverse_diagonal;
    solution[length - 1] = rhs[length - 1];
    index = length - 1;
    while (index-- > 0) {
        solution[index] = rhs[index] - tmp[index] * solution[index + 1];
    }
}

void optimized_momentum_solve_x(const Grid *grid,
                                const SolverConfig *config,
                                const ProblemDefinition *problem,
                                ScalarField *eta,
                                const ScalarField *zeta,
                                const ScalarField *velocity,
                                const ScalarField *pressure_star,
                                const ScalarField *gamma,
                                Direction component,
                                size_t timestep,
                                RealBuffer *scratch)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    Real *tmp = scratch->data;
    Real *rhs = tmp + nx;
    Real *increment = rhs + nx;
    size_t j;
    size_t k;

    assert(scratch->capacity >= 3 * nx);
    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            const size_t offset = grid_index(grid, 0, j, k);
            Real inverse_diagonal;
            size_t i;
            tmp[0] = (Real)0;
            rhs[0] = evaluate_velocity_boundary_increment(
                grid, config, problem, 0, j, k, timestep, component);
            for (i = 1; i + 1 < nx; ++i) {
                const size_t index = offset + i;
                const Real w = -gamma->data[index] *
                               grid->inverse_spacing_square[DIRECTION_X];
                inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)2 * w) - w * tmp[i - 1]);
                tmp[i] = w * inverse_diagonal;
                rhs[i] = evaluate_momentum_x_rhs(
                    grid, config, problem, eta, zeta, velocity,
                    pressure_star, gamma, i, j, k, timestep, component);
                rhs[i] = (rhs[i] - w * rhs[i - 1]) * inverse_diagonal;
            }
            if (component == DIRECTION_X) {
                increment[nx - 1] = evaluate_velocity_boundary_increment(
                    grid, config, problem, nx - 1, j, k, timestep,
                    component);
            } else {
                const size_t index = offset + nx - 1;
                const Real w = -gamma->data[index] *
                               grid->inverse_spacing_square[DIRECTION_X];
                const Real boundary = evaluate_velocity_boundary_increment(
                    grid, config, problem, nx - 1, j, k, timestep,
                    component);
                rhs[nx - 1] = evaluate_momentum_x_rhs(
                    grid, config, problem, eta, zeta, velocity,
                    pressure_star, gamma, nx - 1, j, k, timestep,
                    component) - (Real)2 * w * boundary;
                inverse_diagonal =
                    (Real)1 /
                    (((Real)1 - (Real)3 * w) - w * tmp[nx - 2]);
                rhs[nx - 1] =
                    (rhs[nx - 1] - w * rhs[nx - 2]) * inverse_diagonal;
                increment[nx - 1] = rhs[nx - 1];
            }
            i = nx - 1;
            while (i-- > 0) {
                increment[i] = rhs[i] - tmp[i] * increment[i + 1];
            }
            for (i = 0; i < nx; ++i) {
                eta->data[offset + i] += increment[i];
            }
        }
    }
}

static void scalar_momentum_tail_line(
    const Grid *grid,
    const SolverConfig *config,
    const ProblemDefinition *problem,
    ScalarField *stage,
    ScalarField *rhs_workspace,
    const ScalarField *gamma,
    Direction component,
    size_t timestep,
    Direction direction,
    size_t outer,
    size_t x,
    Real *tmp,
    Real *rhs,
    Real *increment)
{
    const size_t length = grid->extent[direction];
    const size_t stride = grid->stride[direction];
    const size_t offset = direction == DIRECTION_Y
        ? grid_index(grid, x, 0, outer)
        : grid_index(grid, x, outer, 0);
    size_t level;

    tmp[0] = (Real)0;
    rhs[0] = evaluate_velocity_boundary_increment(
        grid, config, problem, x,
        direction == DIRECTION_Y ? 0 : outer,
        direction == DIRECTION_Z ? 0 : outer,
        timestep, component);
    for (level = 1; level + 1 < length; ++level) {
        const size_t index = offset + level * stride;
        const Real w = -gamma->data[index] *
                       grid->inverse_spacing_square[direction];
        const Real inverse_diagonal =
            (Real)1 /
            (((Real)1 - (Real)2 * w) - w * tmp[level - 1]);
        tmp[level] = w * inverse_diagonal;
        rhs[level] = (rhs_workspace->data[index] - w * rhs[level - 1]) *
                     inverse_diagonal;
    }
    {
        const size_t index = offset + (length - 1) * stride;
        const Real boundary = evaluate_velocity_boundary_increment(
            grid, config, problem, x,
            direction == DIRECTION_Y ? length - 1 : outer,
            direction == DIRECTION_Z ? length - 1 : outer,
            timestep, component);
        if (component == direction) {
            increment[length - 1] = boundary;
        } else {
            const Real w = -gamma->data[index] *
                           grid->inverse_spacing_square[direction];
            const Real inverse_diagonal =
                (Real)1 /
                (((Real)1 - (Real)3 * w) - w * tmp[length - 2]);
            rhs[length - 1] =
                rhs_workspace->data[index] - (Real)2 * w * boundary;
            rhs[length - 1] =
                (rhs[length - 1] - w * rhs[length - 2]) *
                inverse_diagonal;
            increment[length - 1] = rhs[length - 1];
        }
    }
    level = length - 1;
    while (level-- > 0) {
        increment[level] = rhs[level] - tmp[level] * increment[level + 1];
    }
    for (level = 0; level < length; ++level) {
        stage->data[offset + level * stride] += increment[level];
    }
}

static void optimized_momentum_directional(
    const Grid *grid,
    const SolverConfig *config,
    const ProblemDefinition *problem,
    const ScalarField *source,
    ScalarField *stage,
    ScalarField *rhs_workspace,
    const ScalarField *gamma,
    Direction component,
    size_t timestep,
    RealBuffer *scratch,
    Direction direction,
    size_t slice_vectors)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t length = grid->extent[direction];
    const size_t stride = grid->stride[direction];
    const size_t outer_count = direction == DIRECTION_Y
        ? grid->extent[DIRECTION_Z]
        : grid->extent[DIRECTION_Y];
    const size_t slice = slice_vectors * SIMD_LENGTH;
    Real *tmp = scratch->data;
    Real *increment = tmp + length * slice;
    Real *boundary = increment + length * slice;
    Real *tail_tmp = boundary + slice;
    Real *tail_rhs = tail_tmp + length;
    Real *tail_increment = tail_rhs + length;
    const SimdReal one = SIMD_SET1((Real)1);
    const SimdReal two = SIMD_SET1((Real)2);
    const SimdReal three = SIMD_SET1((Real)3);
    const SimdReal minus_inverse_h2 =
        SIMD_SET1(-grid->inverse_spacing_square[direction]);
    size_t q;
    size_t outer;

    assert(scratch->capacity >=
           2 * length * slice + slice + 3 * length);
    for (q = 0; q < grid->cell_count; ++q) {
        rhs_workspace->data[q] = source->data[q] - stage->data[q];
    }

    for (outer = 0; outer < outer_count; ++outer) {
        const size_t base = direction == DIRECTION_Y
            ? grid_index(grid, 0, 0, outer)
            : grid_index(grid, 0, outer, 0);
        size_t x = 0;

        for (; x + slice <= nx; x += slice) {
            size_t lane;
            size_t level;
            size_t vector_index;

            for (lane = 0; lane < slice; ++lane) {
                rhs_workspace->data[base + x + lane] =
                    evaluate_velocity_boundary_increment(
                        grid, config, problem, x + lane,
                        direction == DIRECTION_Y ? 0 : outer,
                        direction == DIRECTION_Z ? 0 : outer,
                        timestep, component);
                tmp[lane] = (Real)0;
            }

            for (level = 1; level + 1 < length; ++level) {
                const size_t field_level = base + level * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local =
                        level * slice + vector_index * SIMD_LENGTH;
                    const SimdReal w = SIMD_MUL(
                        SIMD_LOAD(gamma->data + field_index),
                        minus_inverse_h2);
                    const SimdReal inverse_diagonal = SIMD_DIV(
                        one,
                        SIMD_SUB(
                            SIMD_SUB(one, SIMD_MUL(two, w)),
                            SIMD_MUL(w, SIMD_LOAD(tmp + local - slice))));
                    const SimdReal reduced_tmp =
                        SIMD_MUL(w, inverse_diagonal);
                    const SimdReal reduced_rhs = SIMD_MUL(
                        SIMD_SUB(
                            SIMD_LOAD(rhs_workspace->data + field_index),
                            SIMD_MUL(w,
                                SIMD_LOAD(rhs_workspace->data +
                                          field_index - stride))),
                        inverse_diagonal);
                    SIMD_STORE(tmp + local, reduced_tmp);
                    SIMD_STORE(rhs_workspace->data + field_index, reduced_rhs);
                }
            }

            for (lane = 0; lane < slice; ++lane) {
                boundary[lane] = evaluate_velocity_boundary_increment(
                    grid, config, problem, x + lane,
                    direction == DIRECTION_Y ? length - 1 : outer,
                    direction == DIRECTION_Z ? length - 1 : outer,
                    timestep, component);
            }
            if (component == direction) {
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    SIMD_STORE(increment + (length - 1) * slice +
                                   vector_index * SIMD_LENGTH,
                               SIMD_LOAD(boundary +
                                         vector_index * SIMD_LENGTH));
                }
            } else {
                const size_t field_level =
                    base + (length - 1) * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local = (length - 1) * slice +
                                         vector_index * SIMD_LENGTH;
                    const SimdReal w = SIMD_MUL(
                        SIMD_LOAD(gamma->data + field_index),
                        minus_inverse_h2);
                    SimdReal last_rhs = SIMD_SUB(
                        SIMD_LOAD(rhs_workspace->data + field_index),
                        SIMD_MUL(SIMD_MUL(two, w),
                                 SIMD_LOAD(boundary +
                                           vector_index * SIMD_LENGTH)));
                    const SimdReal inverse_diagonal = SIMD_DIV(
                        one,
                        SIMD_SUB(
                            SIMD_SUB(one, SIMD_MUL(three, w)),
                            SIMD_MUL(w, SIMD_LOAD(tmp + local - slice))));
                    last_rhs = SIMD_MUL(
                        SIMD_SUB(last_rhs,
                                 SIMD_MUL(w,
                                     SIMD_LOAD(rhs_workspace->data +
                                               field_index - stride))),
                        inverse_diagonal);
                    SIMD_STORE(increment + local, last_rhs);
                }
            }

            level = length - 1;
            while (level-- > 0) {
                const size_t field_level = base + level * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local =
                        level * slice + vector_index * SIMD_LENGTH;
                    const SimdReal value = SIMD_SUB(
                        SIMD_LOAD(rhs_workspace->data + field_index),
                        SIMD_MUL(SIMD_LOAD(tmp + local),
                                 SIMD_LOAD(increment + local + slice)));
                    SIMD_STORE(increment + local, value);
                }
            }

            for (level = 0; level < length; ++level) {
                const size_t field_level = base + level * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local =
                        level * slice + vector_index * SIMD_LENGTH;
                    SIMD_STORE(stage->data + field_index,
                        SIMD_ADD(SIMD_LOAD(stage->data + field_index),
                                 SIMD_LOAD(increment + local)));
                }
            }
        }

        for (; x < nx; ++x) {
            scalar_momentum_tail_line(
                grid, config, problem, stage, rhs_workspace, gamma,
                component, timestep, direction, outer, x,
                tail_tmp, tail_rhs, tail_increment);
        }
    }
}

void optimized_momentum_solve_y(const Grid *grid,
                                const SolverConfig *config,
                                const ProblemDefinition *problem,
                                const ScalarField *source,
                                ScalarField *stage,
                                ScalarField *rhs_workspace,
                                const ScalarField *gamma,
                                Direction component,
                                size_t timestep,
                                RealBuffer *scratch)
{
    optimized_momentum_directional(grid, config, problem, source, stage,
                                   rhs_workspace, gamma, component, timestep,
                                   scratch, DIRECTION_Y, 4);
}

void optimized_momentum_solve_z(const Grid *grid,
                                const SolverConfig *config,
                                const ProblemDefinition *problem,
                                const ScalarField *source,
                                ScalarField *stage,
                                ScalarField *rhs_workspace,
                                const ScalarField *gamma,
                                Direction component,
                                size_t timestep,
                                RealBuffer *scratch)
{
    optimized_momentum_directional(grid, config, problem, source, stage,
                                   rhs_workspace, gamma, component, timestep,
                                   scratch, DIRECTION_Z, 8);
}

void optimized_pressure_solve_x(const Grid *grid,
                                const SolverConfig *config,
                                const VectorField *velocity,
                                ScalarField *rhs_workspace,
                                ScalarField *psi,
                                RealBuffer *scratch)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t ny = grid->extent[DIRECTION_Y];
    const size_t nz = grid->extent[DIRECTION_Z];
    const Real w = -grid->inverse_spacing_square[DIRECTION_X];
    Real *tmp = scratch->data;
    size_t i;
    size_t j;
    size_t k;

    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            for (i = 0; i < nx; ++i) {
                const size_t index = grid_index(grid, i, j, k);
                if (i == 0 || j == 0 || k == 0) {
                    rhs_workspace->data[index] = (Real)0;
                } else {
                    const Real divergence =
                        (velocity->component[DIRECTION_X].data[index] -
                         velocity->component[DIRECTION_X]
                             .data[index - grid->stride[DIRECTION_X]]) *
                            grid->inverse_spacing[DIRECTION_X] +
                        (velocity->component[DIRECTION_Y].data[index] -
                         velocity->component[DIRECTION_Y]
                             .data[index - grid->stride[DIRECTION_Y]]) *
                            grid->inverse_spacing[DIRECTION_Y] +
                        (velocity->component[DIRECTION_Z].data[index] -
                         velocity->component[DIRECTION_Z]
                             .data[index - grid->stride[DIRECTION_Z]]) *
                            grid->inverse_spacing[DIRECTION_Z];
                    rhs_workspace->data[index] = -divergence / config->dt;
                }
            }
        }
    }
    for (k = 0; k < nz; ++k) {
        for (j = 0; j < ny; ++j) {
            const size_t offset = grid_index(grid, 0, j, k);
            pressure_thomas_line(w, nx, tmp,
                                 rhs_workspace->data + offset,
                                 psi->data + offset);
        }
    }
}

static void optimized_pressure_directional(const Grid *grid,
                                           const ScalarField *input,
                                           ScalarField *output,
                                           RealBuffer *scratch,
                                           Direction direction,
                                           size_t slice_vectors)
{
    const size_t nx = grid->extent[DIRECTION_X];
    const size_t length = grid->extent[direction];
    const size_t stride = grid->stride[direction];
    const size_t outer_count = direction == DIRECTION_Y
        ? grid->extent[DIRECTION_Z]
        : grid->extent[DIRECTION_Y];
    const size_t slice = slice_vectors * SIMD_LENGTH;
    const Real scalar_w = -grid->inverse_spacing_square[direction];
    Real *tmp = scratch->data;
    Real *tail_tmp = tmp + length * slice;
    Real *tail_rhs = tail_tmp + length;
    Real *tail_solution = tail_rhs + length;
    const SimdReal one = SIMD_SET1((Real)1);
    const SimdReal two = SIMD_SET1((Real)2);
    const SimdReal w = SIMD_SET1(scalar_w);
    const SimdReal left_inverse =
        SIMD_DIV(one, SIMD_SUB(one, SIMD_MUL(two, w)));
    const SimdReal left_tmp = SIMD_MUL(SIMD_MUL(two, w), left_inverse);
    size_t outer;

    assert(scratch->capacity >= length * slice + 3 * length);
    for (outer = 0; outer < outer_count; ++outer) {
        const size_t base = direction == DIRECTION_Y
            ? grid_index(grid, 0, 0, outer)
            : grid_index(grid, 0, outer, 0);
        size_t x = 0;

        for (; x + slice <= nx; x += slice) {
            size_t vector_index;
            size_t level;
            for (vector_index = 0;
                 vector_index < slice_vectors;
                 ++vector_index) {
                const size_t field_index = x + vector_index * SIMD_LENGTH;
                SIMD_STORE(output->data + base + field_index,
                    SIMD_MUL(SIMD_LOAD(input->data + base + field_index),
                             left_inverse));
                SIMD_STORE(tmp + vector_index * SIMD_LENGTH, left_tmp);
            }
            for (level = 1; level + 1 < length; ++level) {
                const size_t field_level = base + level * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local =
                        level * slice + vector_index * SIMD_LENGTH;
                    const SimdReal inverse_diagonal = SIMD_DIV(
                        one,
                        SIMD_SUB(SIMD_SUB(one, SIMD_MUL(two, w)),
                                 SIMD_MUL(w,
                                     SIMD_LOAD(tmp + local - slice))));
                    SIMD_STORE(tmp + local,
                               SIMD_MUL(w, inverse_diagonal));
                    SIMD_STORE(output->data + field_index,
                        SIMD_MUL(
                            SIMD_SUB(SIMD_LOAD(input->data + field_index),
                                     SIMD_MUL(w,
                                         SIMD_LOAD(output->data +
                                                   field_index - stride))),
                            inverse_diagonal));
                }
            }
            {
                const size_t field_level =
                    base + (length - 1) * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local = (length - 1) * slice +
                                         vector_index * SIMD_LENGTH;
                    const SimdReal inverse_diagonal = SIMD_DIV(
                        one,
                        SIMD_SUB(SIMD_SUB(one, w),
                                 SIMD_MUL(w,
                                     SIMD_LOAD(tmp + local - slice))));
                    SIMD_STORE(output->data + field_index,
                        SIMD_MUL(
                            SIMD_SUB(SIMD_LOAD(input->data + field_index),
                                     SIMD_MUL(w,
                                         SIMD_LOAD(output->data +
                                                   field_index - stride))),
                            inverse_diagonal));
                }
            }
            level = length - 1;
            while (level-- > 0) {
                const size_t field_level = base + level * stride + x;
                for (vector_index = 0;
                     vector_index < slice_vectors;
                     ++vector_index) {
                    const size_t field_index =
                        field_level + vector_index * SIMD_LENGTH;
                    const size_t local =
                        level * slice + vector_index * SIMD_LENGTH;
                    SIMD_STORE(output->data + field_index,
                        SIMD_SUB(SIMD_LOAD(output->data + field_index),
                                 SIMD_MUL(SIMD_LOAD(tmp + local),
                                          SIMD_LOAD(output->data +
                                                    field_index + stride))));
                }
            }
        }
        for (; x < nx; ++x) {
            size_t level;
            const size_t offset = base + x;
            for (level = 0; level < length; ++level) {
                tail_rhs[level] = input->data[offset + level * stride];
            }
            pressure_thomas_line(scalar_w, length, tail_tmp, tail_rhs,
                                 tail_solution);
            for (level = 0; level < length; ++level) {
                output->data[offset + level * stride] = tail_solution[level];
            }
        }
    }
}

void optimized_pressure_solve_y(const Grid *grid,
                                const ScalarField *input,
                                ScalarField *output,
                                RealBuffer *scratch)
{
    optimized_pressure_directional(grid, input, output, scratch,
                                   DIRECTION_Y, 16);
}

void optimized_pressure_solve_z(const Grid *grid,
                                const ScalarField *input,
                                ScalarField *output,
                                RealBuffer *scratch)
{
    optimized_pressure_directional(grid, input, output, scratch,
                                   DIRECTION_Z, 16);
}

const int optimized_backend_has_explicit_simd = OPTIMIZED_HAS_SIMD;
