#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "types.h"
#include "params.h"
#include "decomp.h"
#include "physics.h"
#include "utils.h"

/*
 * Starting values. They remain selectable at compile time, and now they are
 * the defaults of `sim`, which a configuration file can override at start-up:
 *
 *   cc ... -DDEFAULT_WIDTH=32 -DDEFAULT_T=1.0 -DDEFAULT_STEPS=100
 *   ./solver config.txt
 */
#ifndef DEFAULT_WIDTH
#define DEFAULT_WIDTH 128
#endif

#ifndef DEFAULT_HEIGHT
#define DEFAULT_HEIGHT 128
#endif

#ifndef DEFAULT_DEPTH
#define DEFAULT_DEPTH 128
#endif

// Physical domain
#ifndef DEFAULT_LX
#define DEFAULT_LX M_PI
#endif

#ifndef DEFAULT_LY
#define DEFAULT_LY M_PI
#endif

#ifndef DEFAULT_LZ
#define DEFAULT_LZ M_PI
#endif

#ifndef DEFAULT_T
#define DEFAULT_T 1e-0
#endif

#ifndef DEFAULT_STEPS
#define DEFAULT_STEPS 200
#endif

#ifndef DEFAULT_WR_FREQ
#define DEFAULT_WR_FREQ 5
#endif

// Kinematic viscosity
#ifndef DEFAULT_NU
#define DEFAULT_NU 1.0
#endif

/*
 * The names with which the solver reads the parameters. They used to be
 * compile-time constants, now they are the fields of `sim`: the points of use
 * do not change, the value comes from params.h. The spacings and their
 * inverses are fields and not expressions, so the division is done once at
 * start-up and not inside the loops.
 */
#define WIDTH  (sim.width)
#define HEIGHT (sim.height)
#define DEPTH  (sim.depth)

#define GRID_CELLS ((size_t)WIDTH * (size_t)HEIGHT * (size_t)DEPTH)

#define LX (sim.lx)
#define LY (sim.ly)
#define LZ (sim.lz)

#define DX (sim.dx)
#define DY (sim.dy)
#define DZ (sim.dz)

#define DX_INVERSE (sim.dx_inverse)
#define DY_INVERSE (sim.dy_inverse)
#define DZ_INVERSE (sim.dz_inverse)
#define DX_INVERSE_SQUARE (sim.dx_inverse_square)
#define DY_INVERSE_SQUARE (sim.dy_inverse_square)
#define DZ_INVERSE_SQUARE (sim.dz_inverse_square)

#define T       (sim.t_end)
#define STEPS   (sim.steps)
#define DT      (sim.dt)
#define WR_FREQ (sim.wr_freq)
#define NU      (sim.nu)


typedef struct SolverMemState {
    VectorField eta;
    VectorField zeta;
    VectorField u;
    VectorField k;
    ScalarField pressure;
    ScalarField pressure_star;
    /*
     * The scratch of the tridiagonal backend, opaque to everything else.
     *
     * The two backends need different and incompatible things -- Schur the
     * SIMD buffers and the three already-factorised pressure matrices, the
     * pipelined Thomas its c' and d' over the whole block -- and neither type
     * must get all the way here. backend_init allocates it, backend_free frees
     * it, and in between only whoever knows what it is touches it.
     */
    void *backend;
} SolverMemState;

extern const Data paper_data;

/* Looks up a scenario by name; NULL if it does not exist. */
const Data *data_by_name(const char *name);
void data_print_names(FILE *stream);

void solver_init(const Decomp *decomp,
                 SolverMemState *solver_mem_state,
                 Data *data,
                 const char *data_name);

void solver_solve(const Decomp *decomp,
                  SolverMemState *solver_mem_state,
                  Data *data,
                  SolverStats *solver_stats,
                  int write_enabled);

#endif
