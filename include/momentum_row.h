#ifndef MOMENTUM_ROW_H
#define MOMENTUM_ROW_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "decomp.h"
#include "physics.h"
#include "solver.h"
#include "types.h"

/*
 * The row of the momentum system for a point, and nothing else.
 *
 * This file exists for a single reason: the physics of a point must be written
 * in one place only, even when there are two ways of solving the system. The
 * Schur complement needs the four diagonals written out in full, because it
 * rereads the matrix three times (once for the right-hand side, twice for the
 * influence functions). The pipelined Thomas consumes them on the spot and
 * never writes them. These are two incompatible memory layouts, but the
 * formulas are the same, and they are these.
 *
 * Therefore what is shared is *the row*, not the elimination: whoever
 * assembles puts it in an array, whoever eliminates consumes it in registers.
 *
 * And it is `static inline` and not a function pointer on purpose: this way
 * the row never reaches memory, and whoever consumes it on the spot pays
 * nothing for having asked a shared function for it.
 */

/* a*x[t-1] + b*x[t] + c*x[t+1] = f, with the a of the first point and the c of
 * the last one never read. */
typedef struct MomentumRow {
    Real a, b, c, f;
} MomentumRow;

/*
 * What does not change from one point to another of the same direction and the
 * same component. It is built once outside the loops: inside, the row costs
 * only the computations that really depend on the cell.
 */
typedef struct MomentumLine {
    const Decomp *d;
    const SolverMemState *state;
    const Data *data;
    const Real *k_porosity;  /* permeability of the component */
    const Real *source;      /* the stage one starts from */
    const Real *target;      /* the stage one arrives at (when reading) */
    Real inverse_square;     /* 1/h^2 along this axis */
    /*
     * The physical term g of the line along x, already computed, indexed by
     * the position along the axis -- or NULL.
     *
     * Along x, in the eta step, both backends fill it with g_line and point it
     * here: each thread prepares a line before eliminating it, using a private
     * buffer that it reuses on the next line. The gain is not only the
     * indirect call to the forcing that disappears from the inner loop: on the
     * line the support of g and the choice of the ghost node do not change, so
     * they are decided once and what remains is vectorised.
     *
     * NULL means "I did not prepare it": one falls back to g_value_here cell
     * by cell. On the y and z axes the pointer is not even read, because g
     * does not enter; along x today no caller arrives with NULL.
     */
    const Real *source_term;
    int axis;
    int v_comp;
    int t_step;
    int last_global;         /* last global index along the axis */
    bool same_direction;     /* component normal to the wall */
} MomentumLine;

/*
 * eta starts from u, zeta starts from eta, u starts from zeta: the chain of
 * the three stages of direction splitting.
 */
static inline MomentumLine momentum_line(const Decomp *d,
                                         const SolverMemState *state,
                                         const Data *data,
                                         int t_step, int v_comp, int axis) {
    const VectorField *from = (axis == 0) ? &state->u
                            : (axis == 1) ? &state->eta
                                          : &state->zeta;
    const VectorField *to = (axis == 0) ? &state->eta
                          : (axis == 1) ? &state->zeta
                                        : &state->u;
    MomentumLine line;

    switch (v_comp) {
        case 0:
            line.k_porosity = state->k.v_x;
            line.source = from->v_x;
            line.target = to->v_x;
            break;
        case 1:
            line.k_porosity = state->k.v_y;
            line.source = from->v_y;
            line.target = to->v_y;
            break;
        case 2:
            line.k_porosity = state->k.v_z;
            line.source = from->v_z;
            line.target = to->v_z;
            break;
        default:
            fprintf(stderr, "Value of v_comp doesn't exist");
            exit(1);
    }

    line.d = d;
    line.source_term = NULL;
    line.state = state;
    line.data = data;
    line.inverse_square = (axis == 0) ? (Real)DX_INVERSE_SQUARE
                        : (axis == 1) ? (Real)DY_INVERSE_SQUARE
                                      : (Real)DZ_INVERSE_SQUARE;
    line.axis = axis;
    line.v_comp = v_comp;
    line.t_step = t_step;
    line.last_global = d->n_global[axis] - 1;
    line.same_direction = (v_comp == axis);
    return line;
}

/*
 * The row of point `cell` (local indices), which in the field array is at
 * `here`. The offset is passed by the caller because it already has it: the
 * loops compute it once per line and then increment it, and recomputing it in
 * here with decomp_index would be one wasted multiplication per cell.
 */
static inline MomentumRow momentum_row(const MomentumLine *line,
                                       const int cell[3], size_t here) {
    const Decomp *d = line->d;
    const int axis = line->axis;
    int gi = decomp_global(d, cell[0], 0);
    int gj = decomp_global(d, cell[1], 1);
    int gk = decomp_global(d, cell[2], 2);
    int along = (axis == 0) ? gi : (axis == 1) ? gj : gk;
    MomentumRow row;

    if (along == 0) {
        /* Lower wall of the domain: prescribed value. */
        row.a = 0.0;
        row.b = 1.0;
        row.c = 0.0;
        row.f = bc_left(line->data->bc_velocity, gi, gj, gk,
                        line->t_step, line->v_comp);
        return row;
    }

    Real k_i = line->k_porosity[here];
    Real w_i = -gamma_from_k(k_i) * line->inverse_square;
    Real rhs = line->source[here] - line->target[here];

    /* Only the first step carries the physical term g. */
    if (axis == 0) {
        /* Without the term already prepared one goes through g_value_here,
         * which evaluates the forcing where it already has the coordinates:
         * calling forcing_at_cell and passing its result to g_value would cost
         * the same three global coordinates computed twice, for every cell. */
        Real source = (line->source_term != NULL)
            ? line->source_term[cell[0]]
            : g_value_here(d, cell[0], cell[1], cell[2], line->t_step, k_i,
                           line->state, line->data, line->v_comp);

        rhs += (DT / beta_from_k(k_i)) * source;
    }

    if (along < line->last_global) {
        row.a = w_i;
        row.b = 1.0 - 2.0 * w_i;
        row.c = w_i;
        row.f = rhs;
    } else if (line->same_direction) {
        row.a = 0.0;
        row.b = 1.0;
        row.c = 0.0;
        row.f = bc_right(line->data->bc_velocity, gi, gj, gk,
                         line->t_step, line->v_comp);
    } else {
        /* Upper wall, tangential component: ghost node eliminated with the
         * boundary condition. */
        Real right_value = bc_right(line->data->bc_velocity, gi, gj, gk,
                                    line->t_step, line->v_comp);
        row.a = w_i;
        row.b = 1.0 - 3.0 * w_i;
        row.c = 0.0;
        row.f = rhs - 2.0 * w_i * right_value;
    }
    return row;
}

#endif
