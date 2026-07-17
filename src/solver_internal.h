#ifndef SOLVER_INTERNAL_H
#define SOLVER_INTERNAL_H

#include "solver.h"

bool real_buffer_init(RealBuffer *buffer, size_t capacity);
void real_buffer_destroy(RealBuffer *buffer);
uint64_t solver_time_ns(void);

/* Absolute boundary values are used to initialize/evaluate the physical
 * solution; split momentum systems use the n+1 minus n increment below. */
Real evaluate_velocity_boundary(const ProblemDefinition *problem,
                                Real x,
                                Real y,
                                Real z,
                                Real time,
                                Direction component);
Real evaluate_velocity_boundary_increment(const Grid *grid,
                                          const SolverConfig *config,
                                          const ProblemDefinition *problem,
                                          size_t i,
                                          size_t j,
                                          size_t k,
                                          size_t timestep,
                                          Direction component);

/* Evaluates the baseline momentum source at its staggered space/time levels.
 * Values outside the component-specific support are exactly zero. */
Real evaluate_g(const Grid *grid,
                const SolverConfig *config,
                const ProblemDefinition *problem,
                const ScalarField *eta,
                const ScalarField *zeta,
                const ScalarField *velocity,
                const ScalarField *pressure_star,
                const ScalarField *gamma,
                size_t i,
                size_t j,
                size_t k,
                size_t timestep,
                Direction component);
Real evaluate_momentum_x_rhs(const Grid *grid,
                             const SolverConfig *config,
                             const ProblemDefinition *problem,
                             const ScalarField *eta,
                             const ScalarField *zeta,
                             const ScalarField *velocity,
                             const ScalarField *pressure_star,
                             const ScalarField *gamma,
                             size_t i,
                             size_t j,
                             size_t k,
                             size_t timestep,
                             Direction component);

/* Orchestration contracts: all storage is owned by Solver, all workspace is
 * preallocated, and neither routine may allocate during a timestep. */
void momentum_step(const Grid *grid,
                   const SolverConfig *config,
                   const ProblemDefinition *problem,
                   SolverState *state,
                   const ScalarField *gamma,
                   SolverWorkspace *workspace,
                   SolverStats *stats,
                   size_t timestep);
void pressure_step(const Grid *grid,
                   const SolverConfig *config,
                   SolverState *state,
                   SolverWorkspace *workspace,
                   SolverStats *stats);

/* Consumes the final pressure correction and simultaneously reconstructs the
 * next pressure predictor in pressure_pipeline. */
void pressure_finish_step(const Grid *grid,
                          ScalarField *pressure,
                          ScalarField *pressure_pipeline);

SolverStatus solver_step(Solver *solver, size_t timestep);

/* VTI output is synchronous.  A failed directory/file operation is propagated
 * to solver_solve as SOLVER_OUTPUT_ERROR. */
bool output_writer_init(OutputWriter *writer, const SolverConfig *config);
bool output_writer_write(OutputWriter *writer,
                         const Grid *grid,
                         size_t timestep,
                         Real velocity_time,
                         Real pressure_time,
                         const VectorField *velocity,
                         const ScalarField *pressure);
void output_writer_destroy(OutputWriter *writer);

#endif
