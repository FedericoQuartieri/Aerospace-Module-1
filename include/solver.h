#ifndef SOLVER_H
#define SOLVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Precision is a compile-time choice shared by the solver and its clients. */
#if defined(USE_FLOAT)
typedef float Real;
#else
typedef double Real;
#endif

#define SOLVER_BACKEND_STANDARD 0
#define SOLVER_BACKEND_OPTIMIZED 1

#ifndef SOLVER_BACKEND
#define SOLVER_BACKEND SOLVER_BACKEND_STANDARD
#endif

#if SOLVER_BACKEND != SOLVER_BACKEND_STANDARD && \
    SOLVER_BACKEND != SOLVER_BACKEND_OPTIMIZED
#error "Unsupported SOLVER_BACKEND"
#endif

typedef enum {
    DIRECTION_X = 0,
    DIRECTION_Y = 1,
    DIRECTION_Z = 2,
    DIRECTION_COUNT = 3
} Direction;

typedef struct {
    Real *data;
    size_t count;
} ScalarField;

/* Structure-of-arrays layout: components never share an allocation. */
typedef struct {
    ScalarField component[DIRECTION_COUNT];
} VectorField;

/* All fields use row-major storage with X contiguous.  The spacing follows
 * the staggered baseline formula h = 2L / (2N - 1). */
typedef struct {
    size_t extent[DIRECTION_COUNT];
    size_t stride[DIRECTION_COUNT];
    size_t cell_count;
    Real length[DIRECTION_COUNT];
    Real spacing[DIRECTION_COUNT];
    Real inverse_spacing[DIRECTION_COUNT];
    Real inverse_spacing_square[DIRECTION_COUNT];
} Grid;

typedef struct {
    /* Runtime extents and physical lengths; the command-line driver uses a
     * cubic extent, while API clients may configure each direction. */
    size_t extent[DIRECTION_COUNT];
    Real domain_length[DIRECTION_COUNT];
    Real dt;
    size_t steps;
    Real viscosity;
    /* Zero disables output.  Otherwise a VTI snapshot is written every
     * output_frequency completed timesteps. */
    size_t output_frequency;
    const char *output_directory;
} SolverConfig;

typedef Real (*VectorFunction)(Real x,
                               Real y,
                               Real z,
                               Real time,
                               Direction component);
typedef Real (*ScalarFunction)(Real x, Real y, Real z, Real time);

typedef struct {
    const char *name;
    /* Velocity callbacks receive component-staggered coordinates.  Pressure
     * and permeability callbacks receive pressure-grid coordinates. */
    VectorFunction initial_velocity;
    ScalarFunction initial_pressure;
    VectorFunction boundary_velocity;
    /* The forcing is evaluated at (n + 1/2) dt by the momentum step. */
    VectorFunction forcing;
    ScalarFunction permeability;
} ProblemDefinition;

/* Persistent 13N-value state.  eta, zeta, and velocity are the three momentum
 * stages; pressure_star is also reused by the pressure pipeline. */
typedef struct {
    VectorField eta;
    VectorField zeta;
    VectorField velocity;
    ScalarField pressure;
    ScalarField pressure_star;
} SolverState;

typedef struct {
    Real *data;
    size_t capacity;
} RealBuffer;

typedef struct {
    /* One full-grid ping-pong/RHS field plus backend-sized Thomas scratch. */
    ScalarField field;
    RealBuffer scratch;
} SolverWorkspace;

typedef struct {
    /* Kernel totals are accumulated in nanoseconds and averaged only when
     * printed.  Output is deliberately excluded from timestep_compute_ns. */
    uint64_t init_ns;
    uint64_t momentum_kernel_ns[DIRECTION_COUNT];
    uint64_t pressure_kernel_ns[DIRECTION_COUNT];
    uint64_t momentum_total_ns;
    uint64_t pressure_total_ns;
    uint64_t pressure_update_ns;
    uint64_t timestep_compute_ns;
    uint64_t output_ns;
    size_t completed_steps;
} SolverStats;

typedef struct {
    bool enabled;
    size_t frequency;
    /* Borrowed from SolverConfig and valid for the Solver lifetime. */
    const char *directory;
    /* Preallocated SoA-to-VTK interleaving buffer. */
    RealBuffer pack_buffer;
} OutputWriter;

/* Solver owns every field and buffer.  problem and configuration strings are
 * borrowed and must remain valid until solver_destroy(). */
typedef struct {
    SolverConfig config;
    Grid grid;
    const ProblemDefinition *problem;
    SolverState state;
    ScalarField gamma;
    SolverWorkspace workspace;
    SolverStats stats;
    OutputWriter output;
} Solver;

typedef enum {
    SOLVER_SUCCESS = 0,
    SOLVER_INVALID_CONFIG,
    SOLVER_ALLOCATION_ERROR,
    SOLVER_OUTPUT_ERROR,
    SOLVER_NUMERICAL_ERROR
} SolverStatus;

static inline size_t grid_index(const Grid *grid,
                                size_t i,
                                size_t j,
                                size_t k)
{
    return i + j * grid->stride[DIRECTION_Y]
             + k * grid->stride[DIRECTION_Z];
}

/* Allocation failure follows the project-wide xmalloc policy and terminates;
 * false is reserved for invalid arguments. */
bool scalar_field_init(ScalarField *field, size_t count);
void scalar_field_destroy(ScalarField *field);
void scalar_field_fill(ScalarField *field, Real value);
void scalar_field_copy(ScalarField *destination, const ScalarField *source);

/* Vector components are three independent, 64-byte-aligned SoA allocations. */
bool vector_field_init(VectorField *field, size_t count);
void vector_field_destroy(VectorField *field);
void vector_field_fill(VectorField *field, Real value);

/* Builds the staggered metrics once; X is contiguous in every field. */
bool grid_init(Grid *grid, const SolverConfig *config);
Real grid_pressure_coordinate(const Grid *grid,
                              Direction direction,
                              size_t index);
Real grid_velocity_coordinate(const Grid *grid,
                              Direction coordinate_direction,
                              Direction velocity_component,
                              size_t index);

SolverConfig solver_default_config(void);

/* Allocates and initializes all owned storage.  The baseline startup copies
 * pressure(t=0) into pressure_star and initializes Eta, Zeta, and U at t=0. */
SolverStatus solver_init(Solver *solver,
                         const SolverConfig *config,
                         const ProblemDefinition *problem);

/* Advances exactly config.steps without allocating in the numerical path. */
SolverStatus solver_solve(Solver *solver);

/* Releases partial or complete state and is safe to call more than once. */
void solver_destroy(Solver *solver);
const char *solver_backend_name(void);

/* Reports hierarchical means over completed_steps; output time is separate. */
void solver_print_stats(const Solver *solver, FILE *stream);

#endif
