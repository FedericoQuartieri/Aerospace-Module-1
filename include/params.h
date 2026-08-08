#ifndef PARAMS_H
#define PARAMS_H

#include "types.h"

/*
 * I parametri della simulazione.
 *
 * Erano costanti di compilazione, e cambiare griglia o passo temporale voleva
 * dire ricompilare: e' quello che facevano gli script di convergenza e di
 * scaling, un binario per ogni caso. Ora stanno qui.
 *
 * `sim` nasce con i valori di default, che sono ancora quelli scelti a
 * compilazione (DEFAULT_WIDTH e compagni in solver.h): un programma che non
 * legge nessun file si comporta esattamente come prima.
 *
 * Si scrive una volta all'avvio e poi si legge soltanto. Va riempita prima di
 * decomp_init_*, che e' il primo a chiedere quanto e' grande la griglia.
 *
 * I campi dopo la riga vuota sono ricavati dagli altri, non si leggono da
 * file: params_load li ricalcola dopo ogni lettura.
 */
typedef struct SimParams {
    int width, height, depth;
    Real lx, ly, lz;
    Real t_end;
    int steps;
    Real nu;
    int wr_freq;

    Real dx, dy, dz;
    Real dx_inverse, dy_inverse, dz_inverse;
    Real dx_inverse_square, dy_inverse_square, dz_inverse_square;
    Real dt;
} SimParams;

extern SimParams sim;

/*
 * Legge un file di righe `chiave = valore`, una per riga; le righe vuote e
 * quelle che cominciano per '#' sono commenti. Le chiavi sono i nomi dei campi
 * qui sopra, esclusi i derivati. Una chiave sconosciuta o una riga
 * incomprensibile fermano il programma: un parametro scritto male e' un
 * risultato sbagliato, non un dettaglio.
 */
void params_load(const char *path);

#endif
