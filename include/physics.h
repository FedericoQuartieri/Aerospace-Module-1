#ifndef PHYSICS_H
#define PHYSICS_H
#include "decomp.h"
#include "types.h"

struct SolverMemState;

Real beta_from_k(Real k);
Real gamma_from_k(Real k);

Real time_physical_coord(Real t_step);
Real centered_physical_coord(int index, int component);
Real staggered_physical_coord(int index, int component);

/*
 * The boundary helpers describe positions in physical space only, so their
 * i, j, k arguments are *global* indices and the caller converts.  That also
 * turns their internal face tests (i == 0, i == WIDTH - 1, ...) into tests on
 * the global boundary, which is what they must be once the grid is split.
 */
Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component);
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component);

/*
 * g_value addresses memory as well, so it takes local indices plus the
 * decomposition and derives the global ones itself.
 */
Real g_value(const Decomp *d,
             int i, int j, int k, int t_step, Real k_i,
             const struct SolverMemState *solver_mem_state,
             const Data *data, int component, Real forcing);

/*
 * The same, for whoever does not already have the forcing ready: it computes
 * it in here, where the staggered coordinates have already been derived.
 *
 * It exists because the alternative -- calling forcing_at_cell and passing the
 * result to g_value -- recomputes the three global coordinates twice per cell
 * and adds a call that cannot be inlined. The value produced is the same bit
 * for bit: they are the same operations in the same order.
 */
Real g_value_here(const Decomp *d,
                  int i, int j, int k, int t_step, Real k_i,
                  const struct SolverMemState *solver_mem_state,
                  const Data *data, int component);

/*
 * The abscissae of the cells of a line along x, staggered for the requested
 * component exactly as g_value computes them. They depend neither on j and k
 * nor on the time step, so a single fill serves all the lines of the block.
 */
void forcing_line_coords(const Decomp *d, int component, Real *restrict xs);

/*
 * Fills `out` with the forcing term of the cells of line (j, k), which g_value
 * then receives already prepared.
 *
 * It uses forcing_line_fn if the scenario has it, otherwise it calls
 * forcing_fn one cell at a time: the value produced is the same, only the cost
 * of producing it changes. This way the choice of whether to optimise stays
 * with the scenario, and the core knows nothing about the shape of the
 * forcing.
 *
 * It derives y, z and the time in here with the same operations as g_value:
 * the rules on the half-cell offsets remain written in one place only.
 */
void forcing_fill_line(const Decomp *d, const Data *data,
                       int j, int k, int t_step, int component,
                       const Real *restrict xs, Real *restrict out);

/*
 * The forcing term of ONE cell, with the same half cells and the same time as
 * forcing_fill_line.
 *
 * It is what g_value used to compute internally before the forcing arrived
 * already prepared. Today nobody calls it: along x both backends prepare g per
 * line with g_line, and the cell-by-cell fallback of momentum_row goes through
 * g_value_here, which evaluates the forcing by itself.
 */
Real forcing_at_cell(const Decomp *d, const Data *data,
                     int i, int j, int k, int t_step, int component);

/*
 * g of the whole line (j, k) along x: out[i] is what g_value would answer for
 * cell (i, j, k).
 *
 * The line is the unit that makes g cheap. The support test and the choice
 * between the interior second derivative and the ghost node depend only on j
 * and k, which do not change along the line: they are decided once instead of
 * per cell. What remains is a straight computation on cells that are
 * contiguous in memory -- along x the stride is 1 -- so it vectorises with
 * normal loads, with no gather and no transpositions.
 *
 * The cells that do not have that shape go back to g_value: they are the two
 * at the ends of the line, and all those of the lines that rest on a ghost
 * node, where the value at the boundary depends on the abscissa. They are
 * O(1/n) of the work, and this way that algebra stays written just once.
 */
void g_line(const Decomp *d, const Data *data,
            const struct SolverMemState *solver_mem_state,
            const Real *restrict k_porosity,
            int j, int k, int t_step, int component,
            const Real *restrict xs, Real *restrict out);

#endif
