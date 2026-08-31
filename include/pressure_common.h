#ifndef PRESSURE_COMMON_H
#define PRESSURE_COMMON_H

#include "decomp.h"
#include "solver.h"
#include "types.h"

/*
 * La parte della cascata di pressione che non dipende dal backend.
 *
 * Fra il termine noto e l'aggiornamento finale ci sono tre risoluzioni
 * tridiagonali, ed e' solo quelle che i due backend fanno diversamente.  Il
 * resto -- costruire il termine noto, scrivere la matrice, ricomporre la
 * pressione -- e' la stessa fisica, e sta scritto una volta sola.
 */

/*
 * Assembla -div(u) / DT nei punti di pressione.  Le tre facce inferiori
 * globali (i == 0, j == 0 o k == 0) sono messe a zero.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u);

/*
 * La matrice di una linea lungo `axis`, in tre array di d->n[axis] elementi.
 *
 * A differenza della quantita' di moto, qui la matrice e' la stessa per tutte
 * le linee dell'asse e non cambia mai nel tempo: ogni riga dipende solo dalla
 * posizione globale del suo punto.  E' per questo che il pezzo condiviso e'
 * la linea e non la cella -- chiederla cella per cella vorrebbe dire
 * ricalcolare per ogni punto una cosa che si sa dall'avvio.
 */
void pressure_matrix(const Decomp *restrict d, int axis,
                     Real *restrict a, Real *restrict b, Real *restrict c);

/* p^{n+1} = p^n + phi, e la pressione estrapolata per il passo seguente. */
void update_pressure(const Decomp *restrict d,
                     SolverMemState *restrict solver_mem_state);

#endif
