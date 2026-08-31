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
 * Valori di partenza.  Restano scegliibili a compilazione, e ora sono i default
 * di `sim`, che un file di configurazione puo' sovrascrivere all'avvio:
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
 * I nomi con cui il solutore legge i parametri.  Erano costanti di
 * compilazione, ora sono i campi di `sim`: i punti d'uso non cambiano, il
 * valore arriva da params.h.  Le spaziature e i loro inversi sono campi e non
 * espressioni, cosi' la divisione resta fatta una volta all'avvio e non dentro
 * i cicli.
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
     * Lo scratch del backend tridiagonale, opaco per tutto il resto.
     *
     * I due backend hanno bisogno di cose diverse e incompatibili -- Schur dei
     * buffer SIMD e delle tre matrici della pressione gia' fattorizzate, il
     * pipelined Thomas dei suoi c' e d' su tutto il blocco -- e nessuno dei due
     * tipi deve arrivare fin qui.  Lo alloca backend_init, lo libera
     * backend_free, e in mezzo lo tocca solo chi sa cos'e'.
     */
    void *backend;
} SolverMemState;

extern const Data paper_data;

/* Cerca uno scenario per nome; NULL se non esiste. */
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
