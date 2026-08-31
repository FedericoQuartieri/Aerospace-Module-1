#ifndef BACKEND_H
#define BACKEND_H

#include "decomp.h"
#include "solver.h"
#include "types.h"

/*
 * Quello che il solutore chiede al backend tridiagonale, e nient'altro.
 *
 * Sotto questa linea ci sono due implementazioni, scelte a compilazione con
 * TRIDIAG=schur|pipeline, e ognuna e' padrona del proprio ciclo, del proprio
 * scratch e del proprio ordine delle componenti.  Sopra, solver.c non sa
 * quale delle due ha davanti.
 *
 * La giuntura sta qui e non piu' in basso -- non su "risolvi questo blocco di
 * linee" -- per una ragione precisa.  Una firma del tipo
 *
 *     solve(axis, lines, n_local, a, b, c, f, x)
 *
 * non impone soltanto *cosa* calcolare: impone anche dove metterlo, quante
 * linee alla volta guardarlo, e quando aver finito.  Sono le tre cose che il
 * Thomas pipelined fa diversamente:
 *
 *   - non materializza mai le quattro diagonali, le consuma sul posto;
 *   - vuole tutte le linee dell'asse insieme, non un piano alla volta, o la
 *     pipeline e' tutta riempimento e svuotamento;
 *   - tiene le tre componenti in volo (avanti x,y,z poi indietro z,y,x) per
 *     non svuotarsi fra la passata avanti e quella indietro.
 *
 * Alzando la giuntura al passo direzionale, quelle tre scelte restano dentro
 * il backend, che e' il solo posto dove hanno senso.  Cio' che i due
 * condividono e' la fisica di un punto, e sta in momentum_row.h.
 */

/*
 * Lo scratch del backend: chi ne ha bisogno se lo alloca qui e se lo tiene in
 * SolverMemState.backend, che per il codice condiviso e' un puntatore opaco.
 *
 * Il complemento di Schur ci mette i buffer dei kernel SIMD e le tre matrici
 * della pressione gia' fattorizzate; il pipelined Thomas ci mette i suoi
 * c' e d'.  Nessuno dei due tipi arriva a solver.c, che prima invece doveva
 * conoscere SchurPlan per poterlo dichiarare sullo stack.
 */
void backend_init(const Decomp *d, SolverMemState *solver_mem_state);
void backend_free(SolverMemState *solver_mem_state);

/* Nome del backend compilato, per le statistiche e per i test. */
const char *backend_name(void);

/*
 * I tre sistemi della quantita' di moto di un passo temporale, tutti e tre
 * gli assi e tutte e tre le componenti.
 */
void momentum_step(const Decomp *d, SolverMemState *solver_mem_state,
                   Data *data, int t_step, SolverStats *solver_stats);

/*
 * La cascata di pressione di un passo temporale, piu' l'aggiornamento.
 * `pressure_buffer` e' spazio di lavoro grande come un campo scalare, che il
 * chiamante possiede.
 */
void pressure_step(const Decomp *d, SolverMemState *solver_mem_state,
                   ScalarField *pressure_buffer, SolverStats *solver_stats);

#endif
