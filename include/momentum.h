#ifndef MOMENTUM_H
#define MOMENTUM_H
#include "solver.h"
#include "simd_real.h"
#include "types.h"
#include <stdbool.h>

#if defined(USE_SIMD) && SIMD_AVAILABLE

/* Number of native vectors advanced together in each directional solve. */
#ifndef ZETA_SIMD_VECTORS
#define ZETA_SIMD_VECTORS 4
#endif

#ifndef U_SIMD_VECTORS
#define U_SIMD_VECTORS 8
#endif

#if ZETA_SIMD_VECTORS < 1 || U_SIMD_VECTORS < 1
#error "SIMD vector block sizes must be positive"
#endif

#define ZETA_SIMD_LINES (ZETA_SIMD_VECTORS * SIMD_LANES)
#define U_SIMD_LINES (U_SIMD_VECTORS * SIMD_LANES)
#define MOMENTUM_SIMD_MAX_LINES \
    ((ZETA_SIMD_LINES > U_SIMD_LINES) ? ZETA_SIMD_LINES : U_SIMD_LINES)

#else

#define MOMENTUM_SIMD_MAX_LINES 1

#endif

void momentum_step(const Decomp *decomp,
                   SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data,
                   int t_step,
                   SolverStats *solver_stats);


#if defined(USE_SIMD) && SIMD_AVAILABLE
void update_zeta_simd(const Decomp *d,
                      SolverMemState *solver_mem_state,
                      Real *restrict rhs,
                      Real *restrict tmp,
                      Data *data, int t_step, int v_comp,
                      int simd_lines);
void update_u_simd(const Decomp *d,
                   SolverMemState *solver_mem_state,
                   Real *restrict rhs,
                   Real *restrict tmp,
                   Data *data, int t_step, int v_comp,
                   int simd_lines);
#endif

#endif
