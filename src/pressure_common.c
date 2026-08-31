#include "pressure_common.h"
#include "utils.h"
#include "workers.h"

/*
 * La fisica della pressione che non dipende da come si risolve il sistema.
 *
 * Il termine noto, la matrice di una linea e l'aggiornamento finale sono gli
 * stessi qualunque sia il backend tridiagonale: cambia solo chi risolve, in
 * mezzo.  Per la quantita' di moto il pezzo condiviso e' la riga di una
 * *cella*, perche' gamma segue la permeabilita' punto per punto; qui e' la
 * matrice di una *linea* intera, perche' non dipende ne' dal tempo ne' da
 * quale linea sia.  Sono due forme diverse per una ragione fisica, non per
 * gusto.
 */

/*
 * Divergenza della velocita', divisa per il passo temporale: e' il termine
 * noto del primo dei tre passi della pressione.
 *
 * Sulle tre facce inferiori del dominio vale zero (la velocita' e' a
 * divergenza nulla). Sono facce globali: un processo che non le tocca calcola
 * la divergenza anche li', leggendo la cella precedente dall'anello di
 * contorno.
 */
void compute_div(const Decomp *restrict d,
                 Real *restrict u_div,
                 const VectorField *restrict u) {
    const Real *restrict u_x = u->v_x;
    const Real *restrict u_y = u->v_y;
    const Real *restrict u_z = u->v_z;
    const size_t stride_y = d->stride[1];
    const size_t stride_z = d->stride[2];
    const Real x_factor = -(Real)DX_INVERSE / (Real)DT;
    const Real y_factor = -(Real)DY_INVERSE / (Real)DT;
    const Real z_factor = -(Real)DZ_INVERSE / (Real)DT;

    WORKERS_PARALLEL_FOR(workers_many())
    for (int k = 0; k < d->n[2]; k++) {
        int gk = decomp_global(d, k, 2);

        for (int j = 0; j < d->n[1]; j++) {
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            if (gk == 0 || gj == 0) {
                for (int i = 0; i < d->n[0]; i++) {
                    u_div[row + (size_t)i] = (Real)0;
                }
                continue;
            }

            int i0 = d->is_first[0] ? 1 : 0;
            if (d->is_first[0]) {
                u_div[row] = (Real)0;
            }

            for (int i = i0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;

                u_div[index] =
                    (u_x[index] - u_x[index - 1]) * x_factor +
                    (u_y[index] - u_y[index - stride_y]) * y_factor +
                    (u_z[index] - u_z[index - stride_z]) * z_factor;
            }
        }
    }
}

/*
 * La matrice di una linea della cascata di pressione.
 *
 * E' la stessa per tutte le linee dell'asse e non cambia mai nel tempo: ogni
 * riga dipende solo dalla posizione globale del suo punto.
 *
 * Le righe agli estremi sono quelle della condizione di Neumann omogenea
 * (Lecture 5, pp. 15-16), e spettano solo a chi tocca davvero la parete: un
 * processo in mezzo al dominio usa ovunque la riga interna. Sono asimmetriche
 * perche' lo e' la discretizzazione: a sinistra il nodo fantasma dista due
 * mezze celle, a destra una.
 */
void pressure_matrix(const Decomp *restrict d, int axis,
                            Real *restrict a,
                            Real *restrict b,
                            Real *restrict c) {
    const int length = d->n[axis];
    const int last_global = d->n_global[axis] - 1;
    const Real w = (axis == 0) ? -(Real)DX_INVERSE_SQUARE
                 : (axis == 1) ? -(Real)DY_INVERSE_SQUARE
                               : -(Real)DZ_INVERSE_SQUARE;

    for (int t = 0; t < length; t++) {
        int along = decomp_global(d, t, axis);

        if (along == 0) {
            a[t] = (Real)0;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = (Real)2 * w;
        } else if (along == last_global) {
            a[t] = w;
            b[t] = (Real)1 - w;
            c[t] = (Real)0;
        } else {
            a[t] = w;
            b[t] = (Real)1 - (Real)2 * w;
            c[t] = w;
        }
    }
}

void update_pressure(const Decomp *restrict d,
                            SolverMemState *restrict solver_mem_state) {
    Real *restrict pressure = solver_mem_state->pressure.v;
    Real *restrict phi_high = solver_mem_state->pressure_star.v;

    WORKERS_PARALLEL_FOR(workers_many())
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                Real phi = phi_high[index];
                Real pressure_new = pressure[index] + phi;

                pressure[index] = pressure_new;
                phi_high[index] = pressure_new + phi;
            }
        }
    }
}
