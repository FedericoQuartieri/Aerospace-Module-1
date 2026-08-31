#ifndef BACKEND_PIPELINE_H
#define BACKEND_PIPELINE_H

#include <stddef.h>

#include "decomp.h"
#include "types.h"

/*
 * Lo stato privato del backend pipelined Thomas.  Non esce da
 * src/tridiag/pipeline/.
 *
 * L'idea, in una riga: non si spezza Thomas, si nasconde l'attesa mandando
 * in pipeline tanti sistemi indipendenti.  Su 128^3 ci sono ~16000 linee per
 * direzione, tutte indipendenti; invece di far aspettare il processo 1 mentre
 * il processo 0 lavora sulla sua meta' della stessa linea, si lavora a batch
 * di linee diverse:
 *
 *   proc 0:  [batch 0] [batch 1] [batch 2] ...
 *                    v         v         v      manda (c', d')
 *   proc 1:           [batch 0] [batch 1] [batch 2] ...
 *                              v         v
 *   proc 2:                    [batch 0] [batch 1] ...
 *
 * Dopo il riempimento tutti i processi lavorano insieme, ognuno su un batch
 * diverso.  Il prezzo e' la memoria: per poter fare la sostituzione
 * all'indietro piu' tardi bisogna tenere i c' e d' di tutto il blocco locale,
 * per tutte e tre le componenti.
 *
 * Tre componenti e non una perche' la passata avanti va x, y, z e quella
 * indietro z, y, x: quando l'ultimo processo finisce l'avanti di z puo'
 * iniziare subito l'indietro di z, e la pipeline si gira su se' stessa invece
 * di svuotarsi e riempirsi di nuovo.
 */

#ifndef PIPELINE_BATCH_LINES
#define PIPELINE_BATCH_LINES 64
#endif

typedef struct PipelineBackend {
    int batch_lines;           /* linee per batch                          */
    size_t component_capacity; /* elementi di c'/d' per una componente     */
    Real *c_prime;             /* 3 * component_capacity                   */
    Real *d_prime;             /* 3 * component_capacity                   */
    Real *forward;             /* 2 * batch_lines: i (c', d') di giunzione */
    Real *backward;            /* batch_lines: la soluzione di giunzione   */
    Real *matrix_a[3];         /* la matrice della pressione, una per asse */
    Real *matrix_b[3];
    Real *matrix_c[3];
} PipelineBackend;

/* Quante linee indipendenti ha il blocco locale lungo `axis`. */
static inline size_t pipeline_line_count(const Decomp *d, int axis) {
    if (axis == 0) return (size_t)d->n[1] * (size_t)d->n[2];
    if (axis == 1) return (size_t)d->n[0] * (size_t)d->n[2];
    return (size_t)d->n[0] * (size_t)d->n[1];
}

/*
 * La cella (i, j, k) del punto `level` della linea `line` lungo `axis`.
 * Le linee sono numerate con la direzione piu' vicina in memoria che scorre
 * per prima, cosi' punti adiacenti restano adiacenti.
 */
static inline void pipeline_cell(const Decomp *d, int axis, size_t line,
                                 int level, int cell[3]) {
    if (axis == 0) {
        cell[0] = level;
        cell[1] = (int)(line % (size_t)d->n[1]);
        cell[2] = (int)(line / (size_t)d->n[1]);
    } else if (axis == 1) {
        cell[0] = (int)(line % (size_t)d->n[0]);
        cell[1] = level;
        cell[2] = (int)(line / (size_t)d->n[0]);
    } else {
        cell[0] = (int)(line % (size_t)d->n[0]);
        cell[1] = (int)(line / (size_t)d->n[0]);
        cell[2] = level;
    }
}

/*
 * Dove stanno c' e d' del punto `level` della linea `line` del batch.
 *
 * Il layout cambia con l'asse, ed e' una scelta, non un dettaglio.  Lungo X
 * la linea e' contigua in memoria e conviene tenerla tale.  Lungo Y e Z no:
 * li' si mettono vicine le *linee*, cosi' punti allo stesso livello di linee
 * diverse sono adiacenti.  Siccome le linee sono indipendenti per
 * definizione, e' la disposizione che permette di vettorizzare attraverso le
 * linee invece che lungo la linea -- cioe' anche quando quella direzione e'
 * divisa fra processi, dove il percorso di Schur la vettorizzazione la perde.
 */
static inline size_t pipeline_scratch(const PipelineBackend *backend, int axis,
                                      int component, size_t batch, int level,
                                      int line, int length) {
    size_t base = (size_t)component * backend->component_capacity +
                  batch * (size_t)length * (size_t)backend->batch_lines;

    if (axis == 0) {
        return base + (size_t)line * (size_t)length + (size_t)level;
    }
    return base + (size_t)level * (size_t)backend->batch_lines + (size_t)line;
}

#endif
