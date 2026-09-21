#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#ifdef USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

/* Function of space, time and vector component (0: X, 1: Y, 2: Z). */
typedef Real (*VectorFunction)(Real x, Real y, Real z, Real t, int component);

/* Function of space and time */
typedef Real (*ScalarFunction)(Real x, Real y, Real z, Real t);

/*
 * The forcing term of a whole line along x, in one go.
 *
 * The core calls it once per line instead of once per cell. The indirect call
 * is amortised over `n` cells, but above all it disappears from the inner
 * loop: as long as it is there, the compiler can neither inline nor vectorise,
 * and that is why the eta step is the only one of the three without SIMD
 * kernels.
 *
 * Whoever implements it sees the whole line, so everything that depends only
 * on y, z and t is computed once before the loop. That is where the real gain
 * lies: for a separable forcing such as that of the paper, of the five
 * trigonometric calls per cell one remains.
 *
 * The lines run along x because g enters only the eta step, which is the one
 * along x. If one day it entered the others too, the signature would have to
 * be generalised with an axis.
 *
 * `out` has `n` elements and cell i is at (xs[i], y, z). The abscissae arrive
 * already computed and already staggered for the requested component: the core
 * prepares them with the same operations as the scalar path, so whoever
 * implements this function cannot evaluate it at the wrong point nor change
 * its last bit by rebuilding the coordinates on their own.
 *
 * It may stay NULL. In that case the core falls back to forcing_fn cell by
 * cell and the scenario behaves exactly as before, so scenarios that have
 * nothing to gain do not need to be touched.
 */
typedef void (*VectorLineFunction)(Real *restrict out, const Real *restrict xs,
                                   int n, Real y, Real z, Real t,
                                   int component);

typedef struct Data {
    const char *name;
    VectorFunction bc_velocity;
    VectorFunction forcing_fn;
    /* Line version of the same forcing, or NULL: see VectorLineFunction. */
    VectorLineFunction forcing_line_fn;
    VectorFunction porosity_fn;
    int porosity_time_dependent;
    VectorFunction velocity_fn;
    ScalarFunction pressure_fn;
} Data;

typedef struct ScalarField {
    Real *v;
} ScalarField;

typedef struct VectorField {
    Real *v_x;
    Real *v_y;
    Real *v_z;
} VectorField;

typedef struct SolverStats {
    /* Accumulated execution times, in nanoseconds. */
    uint64_t eta_sys;
    uint64_t zeta_sys;
    uint64_t u_sys;
    uint64_t psi_sys;
    uint64_t phi_low_sys;
    uint64_t phi_high_sys;
    uint64_t pressure_update;
    /*
     * The filling of the permeability, when it depends on time.
     *
     * It was not timed, and for that reason it was invisible: the sum of the
     * stages did not make the total and nobody checked. On 256^3 a quarter of
     * the step was missing with one thread and almost two thirds with
     * fifty-six, and it was what set the ceiling of the speedup while it was
     * being looked for elsewhere.
     */
    uint64_t porosity_fill;
    /*
     * How much of the eta step goes into preparing the physical term g,
     * instead of solving the system.
     *
     * g is not the system: it is the right-hand side, and it has a cost of its
     * own -- a call to the scenario, a three-point stencil per axis, the
     * pressure gradient. As long as it was inside eta_sys there was no way of
     * knowing whether a change to the right-hand side was worthwhile, nor how
     * much was left to gain. Now eta_sys minus this is the pure solver.
     *
     * It is the longest branch, not the sum: each thread times its own lines
     * and keeps the maximum, so the number is comparable with eta_sys, which
     * is wall-clock time. Both backends prepare g per line; the maximum of the
     * per-thread sums measures the work of g without including MPI waits and
     * synchronisations.
     */
    uint64_t momentum_source;
    uint64_t comm_steps; /* communication inside the timed solve, excluding output */
    uint64_t solve_steps;
    uint64_t wr_output;
} SolverStats;

#endif
