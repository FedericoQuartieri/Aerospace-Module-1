#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include "solver.h"

/* Kernel signatures are shared by both compile-time backends.  Inputs marked
 * const must not alias writable stage/workspace fields.  Scratch is allocated
 * once by solver_init and reused for every line solve. */
typedef void MomentumXKernel(const Grid *grid,
                             const SolverConfig *config,
                             const ProblemDefinition *problem,
                             ScalarField *eta,
                             const ScalarField *zeta,
                             const ScalarField *velocity,
                             const ScalarField *pressure_star,
                             const ScalarField *gamma,
                             Direction component,
                             size_t timestep,
                             RealBuffer *scratch);

typedef void MomentumDirectionalKernel(const Grid *grid,
                                       const SolverConfig *config,
                                       const ProblemDefinition *problem,
                                       const ScalarField *source,
                                       ScalarField *stage,
                                       ScalarField *rhs_workspace,
                                       const ScalarField *gamma,
                                       Direction component,
                                       size_t timestep,
                                       RealBuffer *scratch);

typedef void PressureXKernel(const Grid *grid,
                            const SolverConfig *config,
                            const VectorField *velocity,
                            ScalarField *rhs_workspace,
                            ScalarField *psi,
                            RealBuffer *scratch);

typedef void PressureDirectionalKernel(const Grid *grid,
                                       const ScalarField *input,
                                       ScalarField *output,
                                       RealBuffer *scratch);

MomentumXKernel standard_momentum_solve_x;
MomentumXKernel optimized_momentum_solve_x;
MomentumDirectionalKernel standard_momentum_solve_y;
MomentumDirectionalKernel optimized_momentum_solve_y;
MomentumDirectionalKernel standard_momentum_solve_z;
MomentumDirectionalKernel optimized_momentum_solve_z;
PressureXKernel standard_pressure_solve_x;
PressureXKernel optimized_pressure_solve_x;
PressureDirectionalKernel standard_pressure_solve_y;
PressureDirectionalKernel optimized_pressure_solve_y;
PressureDirectionalKernel standard_pressure_solve_z;
PressureDirectionalKernel optimized_pressure_solve_z;

size_t standard_scratch_capacity(const Grid *grid);
size_t optimized_scratch_capacity(const Grid *grid);

/* The public API has no runtime dispatch: these aliases bind one backend into
 * each solver_core_* target at compile time. */
#if SOLVER_BACKEND == SOLVER_BACKEND_STANDARD
#define backend_momentum_solve_x standard_momentum_solve_x
#define backend_momentum_solve_y standard_momentum_solve_y
#define backend_momentum_solve_z standard_momentum_solve_z
#define backend_pressure_solve_x standard_pressure_solve_x
#define backend_pressure_solve_y standard_pressure_solve_y
#define backend_pressure_solve_z standard_pressure_solve_z
#define backend_scratch_capacity standard_scratch_capacity
#define SOLVER_BACKEND_NAME "standard"
#else
#define backend_momentum_solve_x optimized_momentum_solve_x
#define backend_momentum_solve_y optimized_momentum_solve_y
#define backend_momentum_solve_z optimized_momentum_solve_z
#define backend_pressure_solve_x optimized_pressure_solve_x
#define backend_pressure_solve_y optimized_pressure_solve_y
#define backend_pressure_solve_z optimized_pressure_solve_z
#define backend_scratch_capacity optimized_scratch_capacity
#define SOLVER_BACKEND_NAME "optimized"
#endif

#endif
