#ifndef PARALLEL_H
#define PARALLEL_H

#include "decomp.h"
#include "types.h"

/*
 * Strato che isola MPI dal resto del programma.
 *
 * Il solutore non include mai <mpi.h>: chiama queste funzioni e basta. Così
 * tutto il codice parallelo sta in un file solo, e chi legge sa dove
 * guardare. In cambio la versione seriale continua a compilare e a girare
 * come prima, e resta il riferimento con cui confrontare i risultati
 * paralleli.
 *
 * Senza -DUSE_MPI le funzioni qui sotto sono stub che descrivono un processo
 * solo, quindi il programma seriale non paga niente.
 *
 * Per ora dicono soltanto chi siamo e quanti siamo: la griglia non è ancora
 * divisa, perciò lanciare più processi risolve più volte lo stesso problema.
 * La divisione arriva quando decomp_init imparerà a dare a ciascuno la sua
 * fetta.
 */

/* Avvia e chiude MPI. argc e argv possono essere NULL. */
void par_init(int *argc, char ***argv);
void par_finalize(void);

/* Numero di questo processo, da 0 a par_size() - 1. */
int par_rank(void);

/* Quanti processi sono in esecuzione. */
int par_size(void);

/* Valore restituito quando da quella parte non c'è nessun vicino. */
#define PAR_NO_NEIGHBOR (-1)

/*
 * Dispone i processi su una griglia 3D, che è il modo in cui verrà divisa la
 * griglia di calcolo. Mettendo 0 in procs[c] si lascia scegliere a MPI quanti
 * processi mettere lungo quella direzione: procs = {1, 1, 0} dà quindi le
 * fette lungo Z.
 *
 * Va chiamata dopo par_init e prima di decomp_init_mpi.
 */
void par_topology_init(const int procs[3]);

/* Quanti processi ci sono lungo ciascuna direzione. */
void par_dims(int dims[3]);

/* Dove sto io dentro quella griglia di processi. */
void par_coords(int coords[3]);

/*
 * Rank del vicino lungo la direzione `axis` (0 = X, 1 = Y, 2 = Z), con
 * step -1 verso il basso e +1 verso l'alto. Restituisce PAR_NO_NEIGHBOR se
 * da quella parte c'è la parete del dominio.
 */
int par_neighbor(int axis, int step);

/* Somma un intero su tutti i processi e restituisce il totale a tutti. */
long long par_sum_long(long long value);

/* Massimo su tutti i processi, restituito a tutti. */
Real par_max_real(Real value);

/*
 * Manda `count` numeri al vicino in direzione `step` lungo `axis` e ne riceve
 * altrettanti dal vicino dalla parte opposta. Dove il vicino non c'è non
 * succede niente e `recv` resta com'era.
 *
 * Usa MPI_Sendrecv: due processi affacciati si scambiano dati nello stesso
 * istante, e con Send e Recv separate si bloccherebbero a vicenda appena i
 * messaggi superano la soglia oltre la quale MPI smette di bufferizzarli.
 */
void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count);

/*
 * Raccoglie `count` numeri da ognuno dei processi allineati lungo `axis` e
 * consegna a tutti il vettore completo, ordinato per posizione lungo l'asse.
 * `recv` deve avere spazio per count * (processi lungo axis) numeri.
 */
void par_line_allgather(int axis, const Real *send, int count, Real *recv);

/*
 * Riempie le celle di contorno di un campo con i valori dei vicini.
 *
 * Ogni processo tiene un anello di celle in più tutt'intorno al proprio
 * blocco. Non gli appartengono: sono la copia dell'ultima fila di celle del
 * vicino, e servono perché i conti che guardano una cella più in là possano
 * essere fatti anche sul bordo del blocco, senza chiedere niente a nessuno.
 *
 * Va richiamata ogni volta che il campo cambia e sta per essere riletto.
 * Dove il vicino non c'è, l'anello resta com'era: lì c'è la parete del
 * dominio, e le condizioni al contorno la trattano già per conto loro.
 */
void par_exchange_halo(const Decomp *d, Real *field);

#endif
