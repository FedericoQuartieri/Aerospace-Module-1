#ifndef WORKERS_H
#define WORKERS_H

#include <stdbool.h>

/*
 * I thread di calcolo, e il poco che serve per non doverli nominare ovunque.
 *
 * Il parallelismo a memoria condivisa di questo solutore ha una sola forma:
 * le linee lungo cui si risolvono i sistemi tridiagonali sono indipendenti
 * fra loro, quindi si spartiscono fra i thread. Una linea non viene mai
 * spezzata: le sue somme restano nello stesso ordine, e il risultato resta
 * identico cifra per cifra a quello di un thread solo. E' la stessa proprieta'
 * che il complemento di Schur garantisce fra processi, e la si verifica allo
 * stesso modo.
 *
 * MPI resta confinato fuori dalle regioni parallele: le collettive di Schur e
 * lo scambio degli aloni li chiama sempre il thread principale, quindi basta
 * MPI_THREAD_FUNNELED e nessuna implementazione ha bisogno di lock interni.
 *
 * Senza -DUSE_OMP tutto qui dentro descrive un thread solo e le direttive
 * spariscono, cosi' la build seriale resta quella di prima.
 */

#ifdef USE_OMP

#include <omp.h>

#define WORKERS_PRAGMA(x) _Pragma(#x)
/* schedule(static): i piani costano tutti uguale, e la ripartizione fissa
 * rende la spartizione riproducibile da un'esecuzione all'altra. */
#define WORKERS_PARALLEL_FOR(cond) \
    WORKERS_PRAGMA(omp parallel for schedule(static) if (cond))

/*
 * La stessa cosa, ma spartendo due cicli annidati invece di uno.
 *
 * Serve dove il ciclo esterno da solo puo' avere meno iterazioni dei thread.
 * Riempire un campo scorre i piani lungo Z, e quando MPI divide proprio quella
 * direzione un blocco puo' averne una manciata: con 56 thread e 4 piani
 * cinquantadue thread non riceverebbero niente. Collassando i due cicli si
 * spartiscono le righe, che sono sempre abbastanza.
 *
 * I due cicli devono essere perfettamente annidati -- niente fra la graffa del
 * primo e il `for' del secondo -- altrimenti collapse non si applica.
 */
#define WORKERS_PARALLEL_FOR_2(cond) \
    WORKERS_PRAGMA(omp parallel for schedule(static) collapse(2) if (cond))

/*
 * Un team aperto una volta sola, con dentro piu' cicli spartiti.
 *
 * WORKERS_PARALLEL_FOR apre e chiude un team a ogni ciclo, il che va bene
 * quando il ciclo e' uno solo. Dove invece ce ne sono due separati da una
 * chiamata MPI -- assembla, comunica, riporta -- aprirne uno per ciascuno
 * significa aprirne due per piano, e i piani sono centinaia.
 *
 * Con queste il team si apre fuori dal ciclo sui piani e resta vivo: dentro,
 * WORKERS_FOR sparisce le linee e WORKERS_MASTER isola la chiamata MPI a un
 * thread solo, che essendo il team al livello piu' esterno e' lo stesso che
 * ha chiamato MPI_Init. E' esattamente cio' che MPI_THREAD_FUNNELED richiede,
 * quindi il livello di thread-safety non cambia.
 *
 * Attenzione a due asimmetrie di OpenMP, perche' sbagliarle non da' errori di
 * compilazione ma risultati sbagliati:
 *   - WORKERS_FOR ha una barriera implicita in USCITA, non in ingresso;
 *   - WORKERS_MASTER non ha nessuna barriera, ne' prima ne' dopo.
 * Quindi dopo il master serve sempre un WORKERS_BARRIER esplicito, altrimenti
 * gli altri thread iniziano a leggere cio' che il master sta ancora scrivendo.
 */
#define WORKERS_PARALLEL(cond) WORKERS_PRAGMA(omp parallel if (cond))
#define WORKERS_FOR WORKERS_PRAGMA(omp for schedule(static))
#define WORKERS_MASTER WORKERS_PRAGMA(omp master)
#define WORKERS_BARRIER WORKERS_PRAGMA(omp barrier)

static inline int workers_available(void) { return omp_get_max_threads(); }
static inline int workers_id(void) { return omp_get_thread_num(); }

#else

/* Senza OpenMP la condizione non la legge nessuno, ma va comunque consumata:
 * cosi' chi la calcola non deve scusarsi con il compilatore di non usarla. */
#define WORKERS_PARALLEL_FOR(cond) (void)(cond);
#define WORKERS_PARALLEL_FOR_2(cond) (void)(cond);

/* Senza OpenMP il blocco che segue WORKERS_PARALLEL e' un blocco e basta, e
 * i cicli dentro girano in fila: e' gia' il comportamento voluto. */
#define WORKERS_PARALLEL(cond) (void)(cond);
#define WORKERS_FOR
#define WORKERS_MASTER
#define WORKERS_BARRIER

static inline int workers_available(void) { return 1; }
static inline int workers_id(void) { return 0; }

#endif

/* Se il thread e' uno solo non si apre nessuna regione parallela: aprirla
 * costa comunque, e non c'e' niente da spartire. */
static inline bool workers_many(void) {
    return workers_available() > 1;
}

/*
 * Quanti insiemi di array di lavoro servono per `items` iterazioni
 * indipendenti: uno per thread, ma senza sprecarne piu' di quante siano le
 * iterazioni.  `allowed` e' falso dove la comunicazione impone di procedere
 * in fila.
 */
static inline int workers_slots(int allowed, int items) {
    int workers;

    if (!allowed || items < 2) {
        return 1;
    }
    workers = workers_available();
    return workers < items ? workers : items;
}

/* Lo slot di chi chiama, valido anche fuori da una regione parallela. */
static inline int workers_slot(int slots) {
    int id;

    if (slots < 2) {
        return 0;
    }
    id = workers_id();
    return id < slots ? id : slots - 1;
}

#endif
