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
 * La riga del sistema di quantita' di moto per un punto, e nient'altro.
 *
 * Questo file esiste per una ragione sola: la fisica di un punto deve stare
 * scritta in un posto solo, anche quando i modi di risolvere il sistema sono
 * due.  Il complemento di Schur ha bisogno delle quattro diagonali scritte per
 * esteso, perche' rilegge la matrice tre volte (una per il termine noto, due
 * per le funzioni di influenza).  Il Thomas pipelined le consuma sul posto e
 * non le scrive mai.  Sono due layout di memoria incompatibili, ma le formule
 * sono le stesse, e sono queste.
 *
 * Percio' quello che si condivide e' *la riga*, non l'eliminazione: chi
 * assembla la mette in un array, chi elimina la consuma nei registri.
 *
 * Ed e' `static inline` e non un puntatore a funzione apposta: cosi' la riga
 * non arriva mai in memoria, e chi la consuma sul posto non paga niente per
 * averla chiesta a una funzione condivisa.
 */

/* a*x[t-1] + b*x[t] + c*x[t+1] = f, con a del primo punto e c dell'ultimo
 * mai letti. */
typedef struct MomentumRow {
    Real a, b, c, f;
} MomentumRow;

/*
 * Quello che non cambia da un punto all'altro della stessa direzione e della
 * stessa componente.  Si costruisce una volta fuori dai cicli: dentro, la
 * riga costa solo i conti che dipendono davvero dalla cella.
 */
typedef struct MomentumLine {
    const Decomp *d;
    const SolverMemState *state;
    const Data *data;
    const Real *k_porosity;  /* permeabilita' della componente          */
    const Real *source;      /* lo stadio da cui si parte               */
    const Real *target;      /* lo stadio in cui si arriva (in lettura) */
    Real inverse_square;     /* 1/h^2 lungo questo asse                 */
    int axis;
    int v_comp;
    int t_step;
    int last_global;         /* ultimo indice globale lungo l'asse      */
    bool same_direction;     /* componente normale alla parete          */
} MomentumLine;

/*
 * eta parte da u, zeta parte da eta, u parte da zeta: la catena dei tre
 * stadi del direction splitting.
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
 * La riga del punto `cell` (indici locali), che nell'array dei campi sta in
 * `here`.  L'offset lo passa il chiamante perche' lo ha gia': i cicli lo
 * calcolano una volta per linea e poi lo incrementano, e ricalcolarlo qui
 * dentro con decomp_index sarebbe una moltiplicazione per cella buttata.
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
        /* Parete inferiore del dominio: valore imposto. */
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

    /* Solo il primo passo porta il termine fisico g. */
    if (axis == 0) {
        rhs += (DT / beta_from_k(k_i)) *
               g_value(d, cell[0], cell[1], cell[2], line->t_step, k_i,
                       line->state, line->data, line->v_comp);
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
        /* Parete superiore, componente tangente: nodo fantasma eliminato con
         * la condizione al contorno. */
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
