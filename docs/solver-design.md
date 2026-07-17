# Design della nuova architettura del solver

## 1. Stato del documento

Questo documento propone una nuova struttura per il solver Navier–Stokes–Brinkman.
L'obiettivo non è rifattorizzare uno a uno i tipi e i moduli esistenti, ma descrivere
un'architettura più piccola costruita a partire dal flusso numerico effettivo.

Il documento definisce:

- il modello dati;
- l'ownership della memoria;
- il ciclo di vita del solver;
- la separazione delle responsabilità;
- le interfacce delle operazioni numeriche;
- la gestione sincrona dell'output;
- i criteri di semplicità e la politica dei commenti;
- la struttura dei test di correttezza e convergenza.

Le implementazioni dei kernel numerici non fanno parte di questa proposta. Le loro
interfacce sono definite in modo da non impedire future implementazioni SIMD,
cache-blocked o specifiche per una particolare architettura. L'implementazione
corrente, corretta e convergente, costituisce il contratto numerico da preservare.

## 2. Obiettivi

La nuova architettura deve:

1. ridurre i tipi fondamentali a campo scalare e campo vettoriale;
2. rendere esplicita e centralizzata la proprietà della memoria;
3. ridurre `main()` alla configurazione e alle tre operazioni `init`, `solve` e
   `destroy`;
4. separare stato persistente, coefficienti numerici e memoria temporanea;
5. evitare qualsiasi allocazione nel ciclo temporale;
6. mantenere un layout SoA adatto agli accessi unit-stride e alla SIMD;
7. evitare campi globali temporanei quando un buffer può essere riutilizzato;
8. eliminare thread, code e copie complete dei campi dalla gestione dell'output;
9. fornire un backend standard memory-aware e un backend con SIMD esplicita;
10. selezionare il backend a compile-time mantenendo identica l'orchestrazione;
11. raccogliere statistiche confrontabili per kernel, sottosistema e solver;
12. lasciare ai kernel futuri la scelta di blocking, vettorizzazione e traversal;
13. mantenere verificabile il significato degli stadi numerici `Eta`, `Zeta` e
    `U`;
14. minimizzare numero di tipi, livelli di API, stati impliciti e percorsi
    alternativi;
15. rendere correttezza e convergenza proprietà automaticamente verificate.

### 2.1 La semplicità è un requisito

La nuova implementazione deve essere la più semplice che soddisfa il metodo
numerico, il layout di memoria e le misure prestazionali richieste. La semplicità
non è un obiettivo estetico secondario: riduce il numero di invarianti da
ricordare, rende confrontabili i kernel e limita gli errori durante le future
ottimizzazioni.

Il criterio è ispirato alla filosofia espressa da Salvatore Sanfilippo
"Antirez": la complessità va conteggiata quando si aggiunge una funzionalità o
si ottimizza una singola dimensione, e il lavoro di design deve mirare a scrivere
meno codice necessario, non a costruire più infrastruttura. I riferimenti sono
[Writing system software: code comments](https://antirez.com/news/124),
[We are destroying software](https://antirez.com/news/145) e la presentazione
[Running Redis for 8 years](https://antirez.com/misc/PDF_Codemotion2017.pdf).

Regole pratiche:

1. introdurre un tipo solamente se possiede memoria, protegge un invariante o
   rappresenta una semantica che il compilatore deve distinguere;
2. introdurre una funzione solamente se assegna un nome a una responsabilità,
   elimina duplicazione significativa o isola un dettaglio che cambia;
3. non aggiungere genericità per casi futuri non ancora esistenti;
4. preferire flussi lineari, ownership singola, early return e dati passati
   esplicitamente;
5. evitare registry, callback interne, gerarchie di oggetti, factory e dispatch
   runtime quando una chiamata diretta o una macro compile-time risolve il caso;
6. non mantenere due rappresentazioni della stessa informazione senza un
   beneficio di memoria o tempo misurato;
7. una nuova ottimizzazione può aumentare la complessità solo nel proprio
   backend e solo dopo aver mostrato un vantaggio ripetibile;
8. il backend standard resta l'implementazione leggibile del metodo e non deve
   assorbire strutture necessarie soltanto alla SIMD;
9. codice morto, implementazioni precedenti commentate e opzioni mai usate
   devono essere eliminate: la cronologia appartiene al version control;
10. quando due soluzioni hanno prestazioni equivalenti, si sceglie quella con
    meno stato, meno passaggi e meno regole implicite.

Ogni nuova struttura o livello di API deve quindi rispondere a una domanda
concreta: quale duplicazione, errore possibile, ownership ambigua o costo
misurato elimina? Se la risposta non è precisa, l'astrazione non viene aggiunta.

### 2.2 Politica dei commenti

I commenti sono parte del design e devono vivere vicino al codice che
descrivono. Il loro scopo è permettere al lettore di capire il contratto senza
ricostruire ogni dettaglio del kernel e ridurre il numero di fatti da tenere
simultaneamente in memoria.

Sono richiesti:

- un breve commento `DESIGN` all'inizio dei file che implementano momentum,
  pressione e workspace, limitato a flusso, ownership e invarianti;
- un commento di contratto sulle funzioni pubbliche o numericamente rilevanti:
  input, output, modifica in-place, aliasing ammesso e livello temporale;
- commenti "perché" dove l'ordine delle operazioni, il riuso di un buffer, una
  boundary condition o una formula potrebbe sembrare sostituibile;
- commenti didattici prima delle parti matematiche non locali, come il sistema
  di Thomas modificato e lo staggering della pressione;
- brevi commenti guida per separare le fasi di un kernel lungo, quando riducono
  realmente il carico cognitivo.

Devono invece essere evitati:

- commenti che ripetono letteralmente un assegnamento o un incremento;
- `TODO` senza una condizione verificabile o un riferimento a lavoro tracciato;
- codice disabilitato conservato come commento;
- descrizioni di una vecchia implementazione non più presente;
- commenti lunghi usati per giustificare un'interfaccia inutilmente complessa.

Un commento non compensa un nome ambiguo o una responsabilità mal separata. Se
la spiegazione è difficile da scrivere, prima si verifica se il codice può
essere semplificato; il commento documenta la complessità numerica inevitabile,
non quella introdotta accidentalmente dall'architettura.

## 3. Non obiettivi

In questa fase non si intende:

- implementare i nuovi kernel numerici;
- scegliere una dimensione definitiva per i blocchi SIMD;
- imporre NEON, AVX o un'altra ISA nell'API del solver;
- cambiare discretizzazione, condizioni al bordo o schema temporale;
- rendere concorrenti i diversi stadi del metodo;
- introdurre un framework generico per campi con un numero arbitrario di
  componenti.

## 4. Interpretazione del metodo numerico corrente

Il solver applica una fattorizzazione direzionale. Per ogni timestep il momentum
solver aggiorna sequenzialmente tre approssimazioni vettoriali:

1. solve lungo X e aggiornamento in-place di `Eta`;
2. solve lungo Y e aggiornamento in-place di `Zeta`;
3. solve lungo Z e aggiornamento in-place di `U`.

Per ciascuna componente della velocità il flusso logico è:

```text
stato al tempo precedente
        |
        v
solve X: Eta  <- Eta  + delta_x(U, Eta, Zeta, p_star, gamma)
        |
        v
solve Y: Zeta <- Zeta + delta_y(Eta - Zeta, gamma)
        |
        v
solve Z: U    <- U    + delta_z(Zeta - U, gamma)
```

`Eta`, `Zeta` e `U` non sono quindi tre differenti tipi di campo. Sono tre istanze
persistenti dello stesso tipo `VectorField`, con ruoli differenti nella
fattorizzazione.

La forma incrementale coerente con i tre sistemi di Thomas descritti per il
solver e implementati dal codice corrente è:

```text
Xi^(n+1) - U^n = dt/beta * g^(n+1/2)

(I - gamma Dxx)(Eta^(n+1)  - Eta^n)  = Xi^(n+1)  - Eta^n
(I - gamma Dyy)(Zeta^(n+1) - Zeta^n) = Eta^(n+1) - Zeta^n
(I - gamma Dzz)(U^(n+1)    - U^n)    = Zeta^(n+1) - U^n
```

Questa forma è il contratto architetturale: ogni riga corrisponde a uno stadio
che risolve un incremento e aggiorna in-place il proprio campo.

### 4.1 Staggering temporale

Indicando con `n` il tempo intero della velocità in ingresso, un timestep del
solver esegue la transizione `n -> n+1`:

```text
tempo intero                         tempo intermedio
------------                         -----------------
Eta^n, Zeta^n, U^n                   pressure persistente
boundary velocity a n                forcing^(n+1/2)
                                     pressure_star^(n+1/2)
        \                                  /
         \                                /
          +------ momentum solver -------+
                         |
                         v
                  Eta^(n+1), Zeta^(n+1), U^(n+1)
                         |
                         v
                  pressure solver
                         |
                         v
             correction phi^(n+1/2)
                         |
                         v
                  pressure^(n+1/2)
             pressure_star^(n+3/2)
```

Nel momentum step che calcola la velocità a `n+1`, `g` è quindi valutata a
`n+1/2` usando:

- `pressure_star` a `n+1/2`;
- la forzante a `n+1/2`;
- `Eta`, `Zeta` e `U` a `n`;
- i valori assoluti delle boundary conditions della velocità a `n`.

Le boundary conditions del sistema incrementale rappresentano invece la
differenza fra velocità prescritta a `n+1` e a `n`.

All'inizializzazione:

```text
Eta = Zeta = U = velocità iniziale a t = 0
pressure            = pressione iniziale a t = 0
pressure_star       = pressione esatta a t = dt/2
```

Il primo momentum step usa dunque `U^0`, forzante e `pressure_star` a `1/2`, e
produce `U^1`. L'inizializzazione di `pressure` a `t=0` è la politica di startup
dell'implementazione corrente; dal primo aggiornamento in poi il campo segue i
livelli seminteri usati dal pressure-correction scheme.

Il pressure solver usa a sua volta tre solve direzionali:

```text
-div(U) / dt
      |
      v
solve X -> Psi
      |
      v
solve Y -> Phi_lower
      |
      v
solve Z -> pressure_correction
      |
      v
aggiornamento di pressure e pressure_star
```

I tre sistemi scalari sono:

```text
(I - Dxx) Psi                    = -div(U^(n+1)) / dt
(I - Dyy) Phi_lower              = Psi
(I - Dzz) correction^(n+1/2)     = Phi_lower
```

Ogni sistema impone condizioni di Neumann omogenee lungo la propria direzione.
Gli operatori della pressione non dipendono da `gamma`: questo permette di
condividere la struttura del workspace di Thomas con il momentum, ma non i suoi
coefficienti numerici.

Il RHS e i tre risultati intermedi non devono essere residenti
contemporaneamente in quattro campi differenti. Inoltre `pressure_star` ha il
suo ultimo uso nel momentum Dxx: durante il pressure solver il suo storage può
essere riutilizzato come una metà del ping-pong e ripristinato con il predittore
del passo successivo prima di uscire da `pressure_step()`:

```text
workspace     = -div(U) / dt
pressure_star = solve_x(workspace)       /* Psi; il vecchio p_star è dead */
workspace     = solve_y(pressure_star)   /* Phi_lower */
pressure_star = solve_z(workspace)       /* correzione finale */
```

La correzione finale soddisfa la ricorrenza:

```text
pressure^(n+1/2) = pressure^(n-1/2) + correction^(n+1/2)

pressure_star^(n+3/2)
    = pressure^(n+1/2) + correction^(n+1/2)
```

Il secondo assegnamento viene eseguito in-place sul buffer che contiene la
correzione e prepara il predittore usato dal momentum step successivo. Di
conseguenza `pressure` e `pressure_star` sono entrambi stato persistente ai
confini del timestep, ma lo storage di `pressure_star` cambia ruolo solamente
all'interno del pressure solver. L'aggiornamento esistente in
`compute_pressure()` implementa già la ricorrenza; deve essere spostato dietro
un'interfaccia che renda esplicita questa transizione.

Questa lifetime reuse elimina un altro campo full-grid senza introdurre aliasing
fra input e output dello stesso kernel: ogni solve direzionale continua a
leggere da un campo e scrivere nell'altro.

Il percorso ottimizzato corrente calcola già il termine `g` e `Xi` durante il
solve X. La nuova architettura formalizza questa scelta:

- non esiste un `GField` persistente;
- non esiste un campo `Xi` persistente;
- non esiste un campo globale `Delta`;
- l'incremento di una linea o di un blocco viene applicato immediatamente allo
  stadio aggiornato.

Le componenti X, Y e Z sono indipendenti durante ogni solve direzionale. Di
conseguenza un solo campo scalare globale può essere riutilizzato come RHS,
processando una componente alla volta.

### 4.2 Semantica comune di `g` e delle boundary conditions

La scelta del backend non deve modificare la formula numerica. Le operazioni
comuni vengono definite semanticamente una sola volta e possono poi essere
fuse, inline o materializzate dal singolo backend.

Per il solve X del momentum, il contratto dell'implementazione corrente è:

```text
g^(n+1/2)
    = forcing^(n+1/2)
    - grad(pressure_star^(n+1/2))
    - nu/K * U^n
    + nu * (Dxx(Eta^n) + Dyy(Zeta^n) + Dzz(U^n))

Xi^(n+1) = U^n + dt/beta * g^(n+1/2)
rhs_x    = Xi^(n+1) - Eta^n
```

`pressure_star` resta quindi parte necessaria dello stato persistente: viene
letto dal calcolo di `g` durante il solve Dxx.

`g_value()` e il vecchio `compute_g()` costituiscono il riferimento da riprodurre
e contro cui verificare backend standard e ottimizzato.

Esistono due concetti differenti di boundary condition:

1. il valore assoluto `u_bc(t_n)`, usato per costruire i ghost point nei termini
   del Laplaciano contenuti in `g`;
2. l'incremento `u_bc(t_(n+1))-u_bc(t_n)`, eventualmente corretto sulle facce
   inferiori per rispettare il vincolo di divergenza, usato come boundary del
   sistema incrementale di Thomas.

Questi due concetti non devono condividere un nome ambiguo come
`get_boundary_velocity`. L'interfaccia comune proposta è:

```c
/* Valore fisico assoluto fornito dalla definizione del problema. */
Real evaluate_velocity_boundary(const ProblemDefinition *problem,
                                Real x,
                                Real y,
                                Real z,
                                Real time,
                                Direction component);

/* Incremento discreto da imporre al sistema di Thomas. Questa funzione
 * centralizza anche la priorità corrente fra facce, spigoli e vertici. */
Real evaluate_velocity_boundary_increment(const Grid *grid,
                                          const SolverConfig *config,
                                          const ProblemDefinition *problem,
                                          size_t i,
                                          size_t j,
                                          size_t k,
                                          size_t timestep,
                                          Direction component);

/* Valutazione puntuale di g con gestione uniforme di interni e bordi. */
Real evaluate_g(const Grid *grid,
                const SolverConfig *config,
                const ProblemDefinition *problem,
                const ScalarField *eta,
                const ScalarField *zeta,
                const ScalarField *velocity,
                const ScalarField *pressure_star,
                const ScalarField *gamma,
                size_t i,
                size_t j,
                size_t k,
                size_t timestep,
                Direction component);

/* Contratto direttamente utile a Dxx. Può combinare Xi-Eta senza
 * materializzare né g né Xi e senza ripetere divisioni evitabili. */
Real evaluate_momentum_x_rhs(const Grid *grid,
                             const SolverConfig *config,
                             const ProblemDefinition *problem,
                             const ScalarField *eta,
                             const ScalarField *zeta,
                             const ScalarField *velocity,
                             const ScalarField *pressure_star,
                             const ScalarField *gamma,
                             size_t i,
                             size_t j,
                             size_t k,
                             size_t timestep,
                             Direction component);
```

Le dichiarazioni sono un contratto matematico, non obbligano a una chiamata di
funzione per ogni punto. Un kernel può usare versioni `static inline`, portare i
valori invarianti fuori dai loop o calcolare più valori insieme. Eventuali cache
delle boundary conditions appartengono al workspace del backend, non allo stato
fisico del solver.

### 4.3 Backend standard e backend ottimizzato

Sono previsti esattamente due backend:

#### `STANDARD`

- non usa intrinsics SIMD esplicite;
- non materializza `GField`, `Xi` o un `Delta` vettoriale;
- elabora una componente alla volta;
- fonde `g`, `Xi` e RHS nel traversal Dxx, evitando passaggi full-grid
  aggiuntivi;
- struttura Dyy e Dzz con loop sequenziali, accessi contigui nell'inner loop e
  assenza di aliasing, in modo da permettere l'auto-vettorizzazione del
  compilatore;
- usa il workspace preallocato e non esegue allocazioni nei kernel.

La versione standard è quindi un baseline scalare ben progettato, non una copia
del vecchio percorso che materializza tutti i temporanei.

#### `OPTIMIZED`

- usa intrinsics SIMD esplicite dove sono disponibili;
- mantiene lo stesso contratto numerico e lo stesso layout dei campi;
- inizialmente adatta i kernel esistenti:
  `optimize_solve_Dxx_tridiag_blocks()`,
  `optimized_simd_solve_Dyy_tridiag_blocks()` e
  `optimized_simd_solve_Dzz_tridiag_blocks()`;
- per la pressione usa inizialmente `compute_Psi()`,
  `optimize_compute_Phi_lower()` e `optimize_compute_Phi_higher()`;
- può fondere RHS, condizioni al bordo e Thomas quando questo riduce il traffico
  di memoria.

La seguente tabella rappresenta il punto di partenza, non il limite finale:

| Stadio | Backend standard | Backend ottimizzato iniziale |
|---|---|---|
| Momentum Dxx | scalare, RHS fuso | `optimize_solve_Dxx_tridiag_blocks()` |
| Momentum Dyy | sequenziale auto-vettorizzabile | `optimized_simd_solve_Dyy_tridiag_blocks()` |
| Momentum Dzz | sequenziale auto-vettorizzabile | `optimized_simd_solve_Dzz_tridiag_blocks()` |
| Pressione Dxx | Thomas scalare memory-aware | `compute_Psi()` adattato al nuovo workspace |
| Pressione Dyy | sequenziale auto-vettorizzabile | `optimize_compute_Phi_lower()` |
| Pressione Dzz | sequenziale auto-vettorizzabile | `optimize_compute_Phi_higher()` |

### 4.4 Selezione compile-time

Il backend viene selezionato con una macro di compilazione:

```c
#define SOLVER_BACKEND_STANDARD  0
#define SOLVER_BACKEND_OPTIMIZED 1

#ifndef SOLVER_BACKEND
#define SOLVER_BACKEND SOLVER_BACKEND_STANDARD
#endif

#if SOLVER_BACKEND != SOLVER_BACKEND_STANDARD && \
    SOLVER_BACKEND != SOLVER_BACKEND_OPTIMIZED
#error "Unsupported SOLVER_BACKEND"
#endif
```

La selezione avviene fuori dai loop numerici. Non sono necessarie function pointer
nel percorso caldo e non esiste un branch per punto o per linea. Il build system
può produrre due eseguibili confrontabili:

```text
solver_standard
solver_optimized
```

Entrambi usano la stessa `SolverConfig`, lo stesso `ProblemDefinition`, lo stesso
layout e lo stesso formato delle statistiche.

I due target devono usare la stessa precisione e gli stessi flag di ottimizzazione
generali. In particolare, il target standard non disabilita l'auto-vettorizzazione
del compilatore: la differenza è l'assenza di intrinsics SIMD esplicite, non la
compilazione deliberatamente non ottimizzata.

Esempio di configurazione CMake:

```cmake
add_executable(solver_standard ${SOLVER_SOURCES})
target_compile_definitions(solver_standard PRIVATE
    SOLVER_BACKEND=SOLVER_BACKEND_STANDARD)

add_executable(solver_optimized ${SOLVER_SOURCES})
target_compile_definitions(solver_optimized PRIVATE
    SOLVER_BACKEND=SOLVER_BACKEND_OPTIMIZED)
```

### 4.5 Direzioni e futura vettorizzazione

Con il layout row-major corrente, X è contiguo. Questo rende Dxx favorevole alla
cache, ma la ricorrenza di Thomas lungo X impedisce di vettorizzare direttamente
il loop sui punti della singola linea. Per usare SIMD occorre lavorare su più
sistemi X indipendenti, i cui elementi non sono naturalmente adiacenti nello
stesso vettore. Le future opzioni includono packing di più righe, trasposizione a
blocchi o un layout interno specifico del backend.

Per Dyy e Dzz, invece, più sistemi indipendenti corrispondono a posizioni X
adiacenti. Un loop esterno sulla ricorrenza Y/Z e un inner loop sulle X contigue
espone naturalmente parallelismo al compilatore. Per questo motivo il backend
standard deve essere scritto in forma semplice e auto-vettorizzabile prima di
introdurre altre intrinsics esplicite.

La stessa osservazione vale per la pressione. Dxx di velocità e pressione resta
la principale area di ottimizzazione futura e non deve essere vincolata dal
modello pubblico dei campi.

## 5. Principi del modello dati

### 5.1 Il tipo dipende dalla forma, non dal significato fisico

Non vengono introdotti tipi distinti come `Pressure`, `ForceField`, `GField` o
`VelocityField`. Il significato fisico è espresso dal nome del membro che contiene
il campo:

```c
ScalarField pressure;
ScalarField gamma;
VectorField velocity;
VectorField eta;
```

### 5.2 Layout SoA

Un campo vettoriale contiene tre campi scalari indipendenti:

```text
component[X] -> [x0 x1 x2 x3 ...]
component[Y] -> [y0 y1 y2 y3 ...]
component[Z] -> [z0 z1 z2 z3 ...]
```

Non viene usato un layout interleaved del tipo:

```text
[x0 y0 z0 x1 y1 z1 ...]
```

Il layout SoA è coerente con i solve tridiagonali, che elaborano una componente
alla volta, e consente accessi unit-stride lungo X. I tre array vengono allocati
separatamente e allineati indipendentemente. Una singola allocazione contenente
tre slab potrà essere valutata in futuro, ma non fa parte del contratto del tipo.

### 5.3 Selezione della componente fuori dai loop critici

Il livello di orchestrazione usa `VectorField`; il kernel estrae una singola
componente prima del loop:

```c
Real *restrict u = field->component[component].data;

/* Il loop interno opera sul puntatore contiguo, senza lookup della componente. */
```

### 5.4 Una sola radice di ownership

L'istanza `Solver` possiede tutti i campi e tutti i buffer. Le funzioni numeriche
ricevono riferimenti non-owning e non possono allocare o liberare memoria.

## 6. Tipi fondamentali

I frammenti seguenti descrivono le interfacce proposte; non costituiscono ancora
un'implementazione.

```c
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* La precisione resta una scelta di compilazione per permettere specializzazione
 * e vettorizzazione dei kernel. */
typedef DTYPE Real;

typedef enum {
    DIRECTION_X = 0,
    DIRECTION_Y = 1,
    DIRECTION_Z = 2,
    DIRECTION_COUNT = 3
} Direction;

typedef struct {
    Real *data;
    size_t count;
} ScalarField;

typedef struct {
    /* Ogni componente è una differente allocazione SoA allineata. */
    ScalarField component[DIRECTION_COUNT];
} VectorField;
```

Le operazioni di gestione dei campi sono le uniche autorizzate ad allocarne la
memoria:

```c
bool scalar_field_init(ScalarField *field, size_t count);
void scalar_field_destroy(ScalarField *field);
void scalar_field_fill(ScalarField *field, Real value);
void scalar_field_copy(ScalarField *destination,
                       const ScalarField *source);

bool vector_field_init(VectorField *field, size_t count);
void vector_field_destroy(VectorField *field);
void vector_field_fill(VectorField *field, Real value);
```

Decisioni per l'allocatore:

- ciascun `ScalarField` inizia a un indirizzo allineato almeno a 64 byte;
- l'allineamento è un dettaglio dell'implementazione di `field.c`;
- `destroy` deve accettare anche un campo parzialmente inizializzato;
- dopo `destroy`, `data == NULL` e `count == 0`;
- deve essere scelta una sola politica d'errore: ritorno esplicito oppure
  terminazione in `xmalloc`. Le due politiche non devono essere mescolate.

## 7. Griglia e indicizzazione

La griglia viene costruita una volta a partire dalla configurazione. Stride e
coefficienti metrici vengono memorizzati per evitare prodotti e divisioni
ripetuti nei kernel.

```c
typedef struct {
    size_t extent[DIRECTION_COUNT];

    /* Layout row-major con X contiguo:
     * stride[X] = 1, stride[Y] = nx, stride[Z] = nx * ny. */
    size_t stride[DIRECTION_COUNT];
    size_t cell_count;

    Real length[DIRECTION_COUNT];
    Real spacing[DIRECTION_COUNT];
    Real inverse_spacing[DIRECTION_COUNT];
    Real inverse_spacing_square[DIRECTION_COUNT];
} Grid;

static inline size_t grid_index(const Grid *grid,
                                size_t i,
                                size_t j,
                                size_t k)
{
    return i
         + j * grid->stride[DIRECTION_Y]
         + k * grid->stride[DIRECTION_Z];
}
```

La discretizzazione staggered corrente usa lo stesso numero di valori per le tre
componenti, ma trasla di mezzo passo la coordinata associata alla componente. La
costruzione delle coordinate non deve restare duplicata nei kernel:

```c
Real grid_pressure_coordinate(const Grid *grid,
                              Direction direction,
                              size_t index);

Real grid_velocity_coordinate(const Grid *grid,
                              Direction coordinate_direction,
                              Direction velocity_component,
                              size_t index);
```

`grid_velocity_coordinate` applica `spacing / 2` solamente quando direzione della
coordinata e componente coincidono. La formula con cui viene costruito lo spacing
deve preservare l'attuale collocazione dei punti e non va sostituita implicitamente
con `length / (extent - 1)` durante il refactoring.

Le dimensioni diventano valori runtime, mentre precisione e specializzazioni SIMD
restano scelte di compilazione. Questo consente di eseguire test con griglie
diverse senza ricompilare l'intero solver, senza imporre al kernel una particolare
strategia di ottimizzazione. Un backend futuro può comunque fornire kernel
specializzati per dimensioni note a compile-time; il modello dati non deve
richiederlo.

## 8. Configurazione e definizione del problema

```c
typedef struct {
    size_t extent[DIRECTION_COUNT];
    Real domain_length[DIRECTION_COUNT];

    Real dt;
    size_t steps;
    Real viscosity;

    /* zero disabilita completamente l'output */
    size_t output_frequency;
    const char *output_directory;
} SolverConfig;

typedef Real (*VectorFunction)(Real x,
                               Real y,
                               Real z,
                               Real time,
                               Direction component);

typedef Real (*ScalarFunction)(Real x,
                               Real y,
                               Real z,
                               Real time);

typedef struct {
    const char *name;

    /* Valutata a t=0 per inizializzare eta, zeta e velocity. */
    VectorFunction initial_velocity;

    /* Callback di inizializzazione della pressione. Viene valutata a t=0 per
     * pressure e a +dt/2 per pressure_star. */
    ScalarFunction initial_pressure;

    VectorFunction boundary_velocity;
    VectorFunction forcing;

    /* K è assunto statico nel tempo. La callback evita di conservarne
     * necessariamente una copia dopo la costruzione di gamma. */
    ScalarFunction permeability;
} ProblemDefinition;
```

`ProblemDefinition` è read-only e non possiede memoria del solver. La sua durata
deve essere almeno pari a quella dell'istanza `Solver`.

## 9. Stato persistente

```c
typedef struct {
    /* Stadi persistenti della fattorizzazione del momentum. */
    VectorField eta;
    VectorField zeta;
    VectorField velocity;

    /* Dopo lo startup, all'ingresso del passo n -> n+1 contengono
     * rispettivamente p^(n-1/2) e p_star^(n+1/2). */
    ScalarField pressure;
    ScalarField pressure_star;
} SolverState;
```

All'istante iniziale `eta`, `zeta` e `velocity` sono inizializzati con la stessa
condizione iniziale. Successivamente vengono aggiornati in-place e devono
persistere tra timestep.

La proposta conserva solamente `gamma` perché il percorso attivo usa questo campo
nei tre operatori direzionali. Dalle definizioni correnti

```text
beta  = 1 + dt * nu / (2K)
gamma = dt * nu / (2 beta)
```

si ricavano le combinazioni necessarie al momentum solver:

```text
dt / beta  = 2 gamma / nu
nu / K     = nu / gamma - 2 / dt
```

Non è quindi necessario conservare tre campi contenenti informazioni equivalenti.
La trasformazione algebrica e il suo comportamento numerico dovranno essere
verificati con i test di convergenza prima dell'implementazione definitiva.

## 10. Workspace

I kernel vengono eseguiti sequenzialmente. Non serve quindi un workspace per
ogni direzione: un solo buffer lineare, dimensionato per il kernel più esigente
del backend compilato, può essere riutilizzato da tutti gli stadi.

```c
typedef struct {
    Real *data;
    size_t capacity;
} RealBuffer;

typedef struct {
    /* Un solo campo full-grid:
     *
     * momentum: RHS della componente corrente per Dyy e Dzz;
     * pressure: metà del ping-pong; l'altra metà è lo storage ormai dead di
     * state.pressure_star, ripristinato prima della fine del timestep.
     */
    ScalarField field;

    /* Scratch privato del kernel attivo. Il solver conosce solamente la
     * capacità, non la suddivisione interna. */
    RealBuffer scratch;
} SolverWorkspace;

/* Restituisce il massimo numero di Real richiesto da un singolo kernel del
 * backend selezionato. Viene chiamata una volta da solver_init(). */
size_t backend_scratch_capacity(const Grid *grid);
```

Il backend suddivide localmente `scratch.data` in coefficienti ridotti, RHS,
soluzione e boundary mediante offset verificati rispetto a `capacity`. Dimensione
del blocco, vector length e forma dei segmenti restano dettagli privati del file
del kernel. Non diventano campi permanenti del solver finché non esiste un uso
esterno concreto.

Il backend standard può richiedere soltanto lo spazio per una linea di Thomas;
quello ottimizzato può richiedere più linee o un blocco SIMD. Se un futuro Dxx
usa packing o trasposizione, aumenta il valore restituito dalla funzione del
backend e cambia la propria partizione interna: `SolverState` e l'orchestrazione
non cambiano.

Tutta la memoria viene allocata da `solver_init()` sulla base della griglia e
del backend selezionato a compile-time. Nessuna funzione chiamata da
`solver_step()` esegue allocazioni. Il workspace può crescere in futuro per
supportare packing o trasposizione di blocchi Dxx senza modificare `SolverState`.

## 11. Statistiche

Le statistiche hanno una struttura gerarchica: il tempo dei kernel è contenuto
nel tempo del sottosistema, e momentum e pressione sono contenuti nel tempo di
calcolo del timestep. Questo permette sia il confronto end-to-end sia l'analisi
del singolo kernel.

Gli array direzionali sono indicizzati tramite `Direction`, non tramite numeri
magici:

```c
typedef struct {
    /* Misurato una sola volta; non viene diviso per il numero di timestep. */
    uint64_t init_ns;

    /* Tempi cumulativi dei singoli kernel direzionali. */
    uint64_t momentum_kernel_ns[DIRECTION_COUNT];
    uint64_t pressure_kernel_ns[DIRECTION_COUNT];

    /* Tempi cumulativi dei due solver completi. */
    uint64_t momentum_total_ns;
    uint64_t pressure_total_ns;
    uint64_t pressure_update_ns;

    /* Cumulativo dei timestep: solo calcolo, senza output sincrono. */
    uint64_t timestep_compute_ns;

    /* L'output è separato per non alterare il confronto fra backend. */
    uint64_t output_ns;

    size_t completed_steps;
} SolverStats;
```

Per ogni quantità cumulativa relativa al solve viene stampata la media:

```text
mean = cumulative_ns / completed_steps
```

Il report minimo contiene:

- tempo totale di inizializzazione;
- media di ciascun kernel momentum Dxx/Dyy/Dzz;
- media del momentum solver completo;
- media di ciascun kernel pressione Dxx/Dyy/Dzz;
- media del pressure solver completo;
- media del tempo di calcolo totale per timestep;
- tempo di output separato;
- nome del backend compilato.

I timer dei kernel comprendono l'intero contratto dello stadio: costruzione del
RHS o valutazione di `g`, valutazione delle boundary conditions, Thomas e
aggiornamento dello stadio. In questo modo un backend non appare artificialmente
più veloce spostando lavoro fuori dalla regione misurata. Eventuali timer interni
più dettagliati possono essere aggiunti per il tuning, ma non sostituiscono il
tempo end-to-end del kernel.

Per il momentum, il tempo associato a una direzione comprende l'esecuzione sulle
tre componenti X/Y/Z della velocità, come nelle statistiche correnti di
`compute_eta_next`, `compute_zeta_next` e `compute_u_next`. Per la pressione ogni
direzione contiene un solo campo scalare. `momentum_total_ns` comprende le tre
direzioni del momentum; `pressure_total_ns` comprende le tre direzioni e
l'aggiornamento finale di pressione.

La misurazione avviene nel livello di orchestrazione. I kernel non stampano e non
conoscono il formato delle statistiche.

## 12. Output sincrono

La nuova gestione dell'output non usa thread, mutex, condition variable, ring
buffer o snapshot full-grid.

```c
typedef struct {
    bool enabled;
    size_t frequency;

    /* Riferimento non-owning alla stringa della configurazione. */
    const char *directory;

    /* Piccolo buffer persistente usato solo se il formato VTI richiede di
     * interleavare temporaneamente le tre componenti. */
    RealBuffer pack_buffer;
} OutputWriter;

bool output_writer_init(OutputWriter *writer,
                        const SolverConfig *config);

bool output_writer_write(OutputWriter *writer,
                         const Grid *grid,
                         size_t timestep,
                         Real velocity_time,
                         Real pressure_time,
                         const VectorField *velocity,
                         const ScalarField *pressure);

void output_writer_destroy(OutputWriter *writer);
```

Il writer usa direttamente lo stato corrente e termina la scrittura prima che il
timestep successivo possa modificarlo:

```text
solver_step
    |
    v
output_writer_write(U, pressure)   /* chiamata bloccante */
    |
    v
timestep successivo
```

Dopo il passo `n -> n+1`, il file contiene `velocity` al tempo
`(n+1) dt` e `pressure` al tempo `(n+1/2) dt`. Il writer riceve entrambi i tempi
esplicitamente e li registra nei metadati VTI; non deve dedurli dal nome del
file. Se il formato espone un solo `TimeValue`, quello principale è il tempo
della velocità e il tempo semintero della pressione viene aggiunto come campo di
metadati separato.

Il buffer `pack_buffer` viene allocato una volta e riutilizzato per scrivere il
campo vettoriale a chunk. Non viene mai creata una copia completa di velocità e
pressione.

Politica proposta: un errore di scrittura viene restituito a `solver_solve()`, che
termina con uno stato di errore. Se si desidera continuare la simulazione
disabilitando solamente l'output, questa politica può essere resa configurabile,
ma non deve essere implementata implicitamente nel writer.

## 13. Oggetto Solver e ownership

```c
typedef struct {
    SolverConfig config;
    Grid grid;

    /* Riferimento read-only, non posseduto dal solver. */
    const ProblemDefinition *problem;

    SolverState state;

    /* Unico coefficiente spaziale persistente. Un wrapper con un solo campo
     * non aggiungerebbe ownership o invarianti. */
    ScalarField gamma;

    SolverWorkspace workspace;
    SolverStats stats;
    OutputWriter output;
} Solver;
```

Regole di ownership:

- `Solver` possiede `state`, `gamma`, `workspace` e il buffer dell'output;
- `Solver` non possiede `ProblemDefinition` e le stringhe di configurazione;
- nessuna copia per valore di `Solver`, `ScalarField` o `VectorField` è ammessa
  dopo l'inizializzazione;
- le funzioni ricevono sempre puntatori agli aggregati;
- gli input read-only sono dichiarati `const`;
- `solver_destroy()` è l'unico punto di rilascio complessivo e deve poter gestire
  un'inizializzazione parziale.

## 14. API pubblica e main

```c
typedef enum {
    SOLVER_SUCCESS = 0,
    SOLVER_INVALID_CONFIG,
    SOLVER_ALLOCATION_ERROR,
    SOLVER_OUTPUT_ERROR,
    SOLVER_NUMERICAL_ERROR
} SolverStatus;

SolverStatus solver_init(Solver *solver,
                         const SolverConfig *config,
                         const ProblemDefinition *problem);

SolverConfig solver_default_config(void);

SolverStatus solver_solve(Solver *solver);

void solver_destroy(Solver *solver);
```

`main()` sceglie configurazione e problema, ma non conosce i singoli campi:

```c
int main(void)
{
    const SolverConfig config = solver_default_config();
    Solver solver = {0};

    SolverStatus status = solver_init(&solver, &config, &PAPER_PROBLEM);
    if (status != SOLVER_SUCCESS) {
        solver_destroy(&solver);
        return EXIT_FAILURE;
    }

    status = solver_solve(&solver);
    solver_destroy(&solver);

    return status == SOLVER_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## 15. Responsabilità di init, solve e destroy

### 15.1 `solver_init`

Responsabilità:

1. avviare la misura del tempo di inizializzazione;
2. validare configurazione e callback obbligatorie;
3. costruire `Grid` e i coefficienti metrici derivati;
4. allocare tutti i campi persistenti;
5. allocare il campo full-grid di workspace;
6. dimensionare l'unico buffer scratch per il backend compilato;
7. inizializzare `eta`, `zeta` e `velocity` dalla condizione iniziale;
8. inizializzare `pressure` a `t=0` e `pressure_star` con la pressione esatta a
   `+dt/2`;
9. costruire `gamma` dalla permeabilità spaziale statica;
10. inizializzare l'output e il suo piccolo buffer di packing;
11. registrare `stats.init_ns`;
12. lasciare l'oggetto in uno stato valido oppure completamente distruggibile.

Non esegue timestep e non apre file di soluzione.

### 15.2 `solver_solve`

Responsabilità:

1. iterare sui timestep;
2. chiamare `solver_step`;
3. misurare kernel, momentum, pressione e timestep con timer annidati;
4. aggiornare statistiche cumulative e stato di errore;
5. invocare sincronicamente l'output alla frequenza richiesta;
6. stampare o restituire medie calcolate sui timestep completati;
7. restituire il primo errore non recuperabile.

Non alloca e non libera campi.

### 15.3 `solver_destroy`

Responsabilità:

1. distruggere output e buffer temporanei;
2. distruggere `gamma` e stato;
3. azzerare puntatori e capacità;
4. funzionare anche dopo un `solver_init` fallito parzialmente.

## 16. Interfacce numeriche

Questa sezione definisce solamente le dichiarazioni. Le implementazioni e le
ottimizzazioni dei kernel sono lavoro futuro.

### 16.1 Contratto dei backend e dispatch

I due backend esportano funzioni con la stessa firma. I typedef seguenti
definiscono il contratto senza introdurre function pointer nel percorso di
esecuzione:

```c
typedef void MomentumXKernel(const Grid *grid,
                             const SolverConfig *config,
                             const ProblemDefinition *problem,
                             ScalarField *eta,
                             const ScalarField *zeta,
                             const ScalarField *velocity,
                             const ScalarField *pressure_star,
                             const ScalarField *gamma,
                             Direction component,
                             size_t timestep,
                             RealBuffer *scratch);

typedef void MomentumDirectionalKernel(const Grid *grid,
                                       const SolverConfig *config,
                                       const ProblemDefinition *problem,
                                       const ScalarField *source,
                                       ScalarField *stage,
                                       ScalarField *rhs_workspace,
                                       const ScalarField *gamma,
                                       Direction component,
                                       size_t timestep,
                                       RealBuffer *scratch);

typedef void PressureXKernel(const Grid *grid,
                            const SolverConfig *config,
                            const VectorField *velocity,
                            ScalarField *rhs_workspace,
                            ScalarField *psi,
                            RealBuffer *scratch);

typedef void PressureDirectionalKernel(const Grid *grid,
                                       const ScalarField *input,
                                       ScalarField *output,
                                       RealBuffer *scratch);

MomentumXKernel standard_momentum_solve_x;
MomentumXKernel optimized_momentum_solve_x;

MomentumDirectionalKernel standard_momentum_solve_y;
MomentumDirectionalKernel optimized_momentum_solve_y;
MomentumDirectionalKernel standard_momentum_solve_z;
MomentumDirectionalKernel optimized_momentum_solve_z;

PressureXKernel standard_pressure_solve_x;
PressureXKernel optimized_pressure_solve_x;
PressureDirectionalKernel standard_pressure_solve_y;
PressureDirectionalKernel optimized_pressure_solve_y;
PressureDirectionalKernel standard_pressure_solve_z;
PressureDirectionalKernel optimized_pressure_solve_z;
```

Un header interno seleziona simboli concreti a compile-time:

```c
#if SOLVER_BACKEND == SOLVER_BACKEND_STANDARD
    #define backend_momentum_solve_x standard_momentum_solve_x
    #define backend_momentum_solve_y standard_momentum_solve_y
    #define backend_momentum_solve_z standard_momentum_solve_z
    #define backend_pressure_solve_x standard_pressure_solve_x
    #define backend_pressure_solve_y standard_pressure_solve_y
    #define backend_pressure_solve_z standard_pressure_solve_z
    #define SOLVER_BACKEND_NAME "standard"
#elif SOLVER_BACKEND == SOLVER_BACKEND_OPTIMIZED
    #define backend_momentum_solve_x optimized_momentum_solve_x
    #define backend_momentum_solve_y optimized_momentum_solve_y
    #define backend_momentum_solve_z optimized_momentum_solve_z
    #define backend_pressure_solve_x optimized_pressure_solve_x
    #define backend_pressure_solve_y optimized_pressure_solve_y
    #define backend_pressure_solve_z optimized_pressure_solve_z
    #define SOLVER_BACKEND_NAME "optimized"
#endif
```

Il suffisso `optimized` identifica il backend con SIMD esplicita, anche quando il
primo adattatore Dxx è ancora scalare. Questo consente di sostituire in futuro
solo `optimized_momentum_solve_x` e `optimized_pressure_solve_x` senza modificare
il solver.

### 16.2 Orchestrazione di un timestep

```c
SolverStatus solver_step(Solver *solver, size_t timestep);

void momentum_step(const Grid *grid,
                   const SolverConfig *config,
                   const ProblemDefinition *problem,
                   SolverState *state,
                   const ScalarField *gamma,
                   SolverWorkspace *workspace,
                   SolverStats *stats,
                   size_t timestep);

void pressure_step(const Grid *grid,
                   const SolverConfig *config,
                   SolverState *state,
                   SolverWorkspace *workspace,
                   SolverStats *stats);
```

`momentum_step` e `pressure_step` chiamano esclusivamente i simboli
`backend_*` e misurano il tempo intorno a ogni chiamata. Non contengono `#if`
sparsi nella logica numerica.

### 16.3 Momentum solver

Le componenti vengono elaborate una alla volta. `workspace->field` è il solo
RHS full-grid disponibile per Y e Z. Il solve X riceve direttamente tutti i campi
necessari a valutare `g` e costruisce il proprio RHS durante il traversal di
Thomas.

Per Y e Z il contratto passa sia il campo sorgente sia lo stadio aggiornato:

```text
Y: source = eta,  stage = zeta
Z: source = zeta, stage = velocity
```

Il backend è responsabile dell'intero stadio, compresa la costruzione di
`source-stage`. La prima implementazione può scrivere questa differenza in
`rhs_workspace`; una futura implementazione può fonderla nel forward pass e non
usare affatto il campo full-grid. In entrambi i casi la regione temporizzata è la
stessa.

Il contratto di ogni backend è un aggiornamento completo dello stadio:

```text
solve X: eta     += incremento_x
solve Y: zeta    += incremento_y
solve Z: velocity += incremento_z
```

Questa scelta nasconde una differenza dell'implementazione corrente: i vecchi
kernel standard scrivono prima in `Delta`, mentre quelli ottimizzati sommano
direttamente l'incremento. Nella nuova architettura nessun backend richiede un
`Delta` full-grid; il backend standard applica anch'esso l'incremento non appena
la linea o il blocco sono risolti.

### 16.4 Pressure solver

```c
void pressure_finish_step(const Grid *grid,
                          ScalarField *pressure,
                          ScalarField *pressure_pipeline);
```

Il contratto della funzione, all'uscita del passo `n -> n+1`, è:

```text
pressure      : p^(n-1/2)      -> p^(n+1/2)
pressure_pipeline:
    correction^(n+1/2)         -> p_star^(n+3/2)
```

La prima invocazione costituisce lo startup già descritto: `pressure` entra con
il valore a `t=0`; dalle invocazioni successive vale la notazione semintera del
contratto.

L'aggiornamento dei due campi deve avvenire nello stesso traversal unit-stride:
prima si calcola la nuova pressione e subito dopo il predittore successivo. Non
serve un ulteriore campo full-grid.

```c
/* pressure_pipeline contiene la correzione all'ingresso e viene convertito
 * in-place nel pressure_star del passo successivo. */
for (size_t q = 0; q < grid->cell_count; ++q) {
    const Real correction = pressure_pipeline->data[q];
    const Real pressure_next = pressure->data[q] + correction;

    pressure->data[q] = pressure_next;
    pressure_pipeline->data[q] = pressure_next + correction;
}
```

Il pressure solver associa nomi numerici ai due storage solamente durante la
fase corrente, alternando input e output:

```c
ScalarField *alternate        = &workspace->field;
ScalarField *pressure_pipeline = &state->pressure_star;

/* pressure_pipeline: p_star -> Psi */
backend_pressure_solve_x(..., alternate, pressure_pipeline, ...);

/* alternate: RHS -> Phi_lower */
backend_pressure_solve_y(..., pressure_pipeline, alternate, ...);

/* pressure_pipeline: Psi -> correction */
backend_pressure_solve_z(..., alternate, pressure_pipeline, ...);

/* pressure_pipeline: correction -> p_star del passo successivo */
pressure_finish_step(grid, &state->pressure, pressure_pipeline);
```

Il kernel pressione X riceve direttamente `velocity`, costruisce
`-div(velocity)/dt` in `rhs` e produce `psi`. In questo modo il suo tempo comprende
la preparazione del termine noto, coerentemente con il comportamento corrente di
`compute_Psi()`. I kernel Y e Z ricevono invece un input già definito e producono
il successivo campo ping-pong.

La dichiarazione con input `const` non impone al kernel di conservare una copia
aggiuntiva del RHS: il forward pass può scrivere i valori ridotti direttamente
nel campo di output o nello scratch lineare. In questo modo il contratto non
obbliga i backend futuri a supportare aliasing input/output.

Non esistono quindi strutture persistenti chiamate `Psi`, `PhiLower` o
`PhiHigher`. La temporanea variazione di ruolo di `pressure_star` è confinata in
`pressure_step()`; tutte le altre funzioni vedono il campo solamente nel suo
significato fisico di predittore.

## 17. Struttura proposta dei moduli

```text
include/
`-- solver.h          tipi condivisi e API init/solve/destroy

src/
|-- solver_internal.h dichiarazioni usate soltanto fra moduli interni
|-- kernels.h         contratti dei kernel e selezione compile-time
|-- field.c           allocazione dei campi
|-- grid.c            griglia, coordinate e indicizzazione
|-- solver.c          lifecycle e loop temporale
|-- momentum.c        orchestrazione comune del momentum
|-- pressure.c        orchestrazione comune della pressione
|-- kernels_standard.c
|-- kernels_optimized.c
|-- physics.c         g e boundary conditions comuni
|-- output.c          writer VTI sincrono
`-- main.c
```

`solver.h` è l'unico header pubblico perché il progetto produce un solver, non
un framework di campi generico. Contiene `Real`, `Direction`, configurazione,
definizione del problema, stato necessario e API. `solver_internal.h` raccoglie
le poche dichiarazioni condivise fra i file `.c`; `kernels.h` isola le firme
numeriche e gli alias `backend_*`.

Dipendenze previste:

```text
main -> solver.h
solver -> field, grid, momentum, pressure, output
momentum/pressure -> kernels.h -> backend compilato
kernel standard/optimized -> physics, grid
```

I kernel non includono il livello di orchestrazione e l'output non conosce i
kernel. Non servono header pubblici separati per ogni struttura né un file per
ogni funzione. Se in futuro `kernels_optimized.c` diventa difficile da leggere,
può essere separato per sottosistema o direzione in quel momento, sulla base di
una complessità reale e non prevista in anticipo.

Il build può compilare entrambi i file backend per mantenere disponibili test
diretti dei kernel, ma `kernels.h` espone al solver un solo insieme di simboli
attivi. CMake esclude il backend inattivo dal target di produzione.

## 18. Modello di memoria atteso

Indicando con `N` il numero di punti e trascurando il buffer scratch lineare:

### Stato persistente

- `eta`, `zeta`, `velocity`: `9N` valori;
- `pressure`, `pressure_star`: `2N` valori;
- `gamma`: `N` valori.

Totale persistente: `12N` valori `Real`.

### Workspace full-grid

- un campo condiviso fra RHS momentum e ping-pong pressione: `N` valori.

Totale principale: `13N` valori `Real`.

Con la griglia corrente `128^3` e `Real == double`, un campo scalare occupa circa
16 MiB. Il modello proposto richiede quindi circa 208 MiB più lo scratch del
backend e il piccolo buffer di output.

La struttura corrente arriva invece a circa `29N` valori allocati prima dei
buffer temporanei del pressure solver; l'output asincrono aggiunge inoltre otto
snapshot da `4N` valori ciascuno. Il writer sincrono elimina interamente questi
`32N` valori aggiuntivi.

## 19. Vincoli per le future ottimizzazioni

L'architettura deve preservare i seguenti vincoli:

1. X è la dimensione contigua in memoria;
2. ogni componente vettoriale è uno stream SoA indipendente;
3. ogni componente inizia a un indirizzo opportunamente allineato;
4. i kernel ricevono puntatori contigui e possono dichiarare l'assenza di aliasing
   tramite `restrict` nelle proprie variabili locali o interfacce specializzate;
5. tutti i buffer sono allocati prima del primo timestep;
6. dimensione del blocco e vector length non fanno parte del modello fisico;
7. il workspace espone capacità, non impone un algoritmo;
8. il loop sui componenti resta esterno al loop numerico più interno;
9. l'output non modifica e non copia interamente lo stato;
10. eventuali trasposizioni o layout alternativi devono essere introdotti come
    backend dei kernel, non modificando il contratto pubblico del solver;
11. il backend standard non usa intrinsics e presenta loop semplici al
    compilatore;
12. per Dyy e Dzz l'inner loop attraversa sistemi indipendenti lungo X, favorendo
    l'auto-vettorizzazione;
13. packing o trasposizioni necessarie alla SIMD Dxx restano locali al backend
    ottimizzato;
14. standard e ottimizzato devono produrre risultati numericamente equivalenti e
    attraversare le stesse regole per `g` e boundary conditions.

### 19.1 Priorità delle ottimizzazioni future

La prima priorità è Dxx per momentum e pressione. La dipendenza di Thomas è
lungo la dimensione contigua e impedisce la SIMD sul singolo sistema. Le
ottimizzazioni dovranno quindi valutare parallelismo fra sistemi diversi, costo
del packing, riuso dei coefficienti e traffico aggiuntivo introdotto da eventuali
trasposizioni.

Per Dyy e Dzz la priorità iniziale è invece scrivere un backend standard con:

- inner loop X unit-stride;
- bound e stride semplici;
- puntatori `restrict` estratti prima dei loop;
- assenza di chiamate non inline nell'inner loop;
- blocchi sufficientemente regolari da essere riconosciuti dal compilatore.

Le intrinsics correnti rimangono il backend ottimizzato di riferimento, ma devono
essere confrontate con il codice standard auto-vettorizzato prima di aggiungere
ulteriore complessità.

## 20. Strategia dei test

I test devono verificare proprietà numeriche, non ripetere manualmente il
contenuto di `main()`. Devono usare la stessa API pubblica
`solver_init()`/`solver_solve()`/`solver_destroy()` utilizzata dall'eseguibile e
devono poter cambiare griglia e timestep a runtime.

### 20.1 Problemi della struttura corrente

I test attuali contengono informazioni utili, ma la loro organizzazione non deve
essere mantenuta:

- le stesse fasi di allocazione, inizializzazione, solve, costruzione della
  soluzione esatta e cleanup sono duplicate in più file;
- le soluzioni manifatturate sono ripetute fra test singoli e test di
  convergenza;
- griglia e timestep dipendono da macro globali, quindi uno studio di
  raffinamento richiede più compilazioni o modifiche manuali;
- i test di correttezza controllano principalmente l'errore L2 della velocità e
  possono ignorare pressione, norma infinito e divergenza;
- `test_convergence` stampa un record, ma non verifica l'ordine osservato e può
  terminare con successo anche se il solve fallisce;
- alcuni casi sono selezionati commentando o decommentando codice;
- la generazione VTI è mescolata alla verifica numerica.

La nuova struttura deve rimuovere questa duplicazione invece di nasconderla
dietro un framework di test generico.

### 20.2 Struttura minima dei file

```text
test/
|-- manufactured_cases.h   dichiarazioni dei casi disponibili
|-- manufactured_cases.c   soluzione esatta, forcing, BC, K e dominio
|-- test_support.h          norme, report e una sola funzione di esecuzione
|-- test_support.c
|-- test_correctness.c      correttezza end-to-end a risoluzione fissata
|-- test_convergence.c      raffinamento spaziale e temporale
`-- test_kernel_equivalence.c  confronto diretto standard/ottimizzato
```

Non deve esistere un eseguibile quasi identico per ogni soluzione manifatturata.
I casi sono elementi di una tabella e gli stessi runner li attraversano tutti.
Un caso non ancora valido viene rimosso dalla tabella o marcato esplicitamente
come non registrato nel build; non viene lasciato come blocco commentato.

Le sole strutture test-specifiche necessarie sono:

```c
typedef struct {
    Real l2;
    Real linf;
} ErrorNorm;

typedef struct {
    ProblemDefinition problem;
    SolverConfig base_config;

    VectorFunction exact_velocity;
    ScalarFunction exact_pressure;

    /* Limiti di regressione per la configurazione di correttezza. */
    Real max_velocity_l2;
    Real max_velocity_linf;
    Real max_pressure_l2;
    Real max_pressure_linf;
    Real max_divergence_l2;

    /* Ordini minimi accettati negli studi dedicati. */
    Real min_velocity_space_order;
    Real min_pressure_space_order;
    Real min_velocity_time_order;
    Real min_pressure_time_order;
} ManufacturedCase;

typedef struct {
    ErrorNorm velocity[DIRECTION_COUNT];
    ErrorNorm pressure;
    Real divergence_l2;
} ErrorReport;

bool run_manufactured_case(const ManufacturedCase *test_case,
                           const SolverConfig *config,
                           ErrorReport *report);
```

`run_manufactured_case()` è l'unico punto che crea e distrugge un `Solver`,
disabilita l'output, esegue il solve e calcola gli errori. Usa un solo percorso di
cleanup e restituisce `false` per qualsiasi errore di inizializzazione, solve,
valore non finito o calcolo delle norme. I test chiamanti decidono solamente
quali casi eseguire e quali proprietà asserire. Prima del primo timestep verifica
anche i livelli temporali dei campi inizializzati.

Non è necessario allocare una seconda soluzione completa. Le funzioni esatte
possono essere valutate mentre si attraversano i campi numerici. Per la pressione
servono due passaggi: il primo calcola l'offset medio fra soluzione numerica ed
esatta, il secondo calcola le norme dopo aver sottratto tale offset. Questo rende
il confronto invariante rispetto alla costante arbitraria della pressione senza
modificare lo stato del solver.

Le coordinate sono ottenute dalle stesse funzioni `grid_pressure_coordinate()` e
`grid_velocity_coordinate()` usate dal solver. I test non devono ricostruire a
parte lo staggered grid con formule duplicate.

Le norme devono avere una definizione unica in `test_support.c`:

```text
L2_h   = sqrt(dx * dy * dz * sum(error[q]^2))
Linf_h = max(abs(error[q]))
```

Per la velocità vengono mantenute le norme delle tre componenti e, quando serve
un singolo valore per l'ordine, si usa
`sqrt(L2_x^2 + L2_y^2 + L2_z^2)`.

### 20.3 Casi manifatturati

Ogni caso deve tenere nello stesso file:

- soluzione esatta di velocità e pressione;
- forcing analiticamente coerente;
- boundary conditions;
- permeabilità spaziale;
- dominio e configurazione base;
- ordine atteso e soglie di regressione.

La vicinanza di queste informazioni elimina l'attuale rischio di cambiare una
soluzione senza aggiornare forcing, dominio o `K`. Il set minimo comprende:

1. un caso completo con pressione non nulla e `K` costante, usato come test
   principale di correttezza e convergenza;
2. un caso a pressione nulla, che isola il momentum e rileva dipendenze spurie
   dal pressure predictor;
3. un caso con `K(x,y,z)` non costante, necessario a verificare la costruzione di
   `gamma` e il termine di Brinkman.

Il terzo caso viene registrato solo quando forcing e soluzione esatta sono
coerenti con la stessa permeabilità. Non si mantiene una formula che dichiara
`K(x,y,z)` ma usa internamente `K=1`.

### 20.4 Test di correttezza

`test_correctness` esegue ogni caso su una configurazione piccola ma
rappresentativa, con output disabilitato. Per entrambi i backend verifica:

1. successo di init e solve;
2. assenza di `NaN` e infinito in tutti i campi persistenti;
3. errore L2 e Linf di ogni componente della velocità;
4. errore L2 e Linf della pressione dopo la rimozione dell'offset medio;
5. norma L2 della divergenza finale;
6. rispetto dei valori prescritti sulle boundary conditions;
7. numero di timestep completati e coerenza dei livelli temporali finali;
8. subito dopo init, `Eta`, `Zeta` e `U` a `t=0`, `pressure` a `t=0` e
   `pressure_star` a `dt/2`.

Le soglie non devono essere numeri locali come `1e-3 /* adjust */`. Vengono
registrate nel `ManufacturedCase`, con un nome e una configurazione precisa, e
sono fissate a partire dai risultati del solver di riferimento. Devono essere
abbastanza larghe da tollerare le differenze di arrotondamento, ma abbastanza
strette da fallire in presenza di un kernel, una boundary condition o uno
staggering errato. Pressione e velocità devono entrambe contribuire al pass/fail.

Il test non scrive VTI e non misura prestazioni. Un'opzione esplicita
`--report <file>` può produrre un record JSON diagnostico, ma il codice di uscita
dipende sempre dalle asserzioni e non dalla riuscita della scrittura del report.

### 20.5 Test di convergenza

La correttezza a una singola risoluzione non dimostra la convergenza. Il test di
convergenza esegue almeno tre livelli e calcola per ogni coppia consecutiva:

```text
order = log(error_coarse / error_fine)
      / log(scale_coarse / scale_fine)
```

Lo studio deve mantenere fisso il tempo finale e distinguere due modalità:

- **spaziale**: raffina le tre dimensioni e sceglie `dt` abbastanza piccolo da
  rendere trascurabile l'errore temporale; per uno schema del secondo ordine si
  può usare inizialmente `dt = C h^2`;
- **temporale**: mantiene una griglia sufficientemente fine e dimezza `dt`,
  evitando livelli nei quali l'errore spaziale ha già prodotto un plateau.

Per ogni livello vengono registrati `h`, `dt`, errori L2/Linf delle tre velocità,
errore di pressione e divergenza. Il test fallisce se:

- un solve fallisce o produce valori non finiti;
- l'errore cresce durante due raffinamenti consecutivi;
- l'ordine L2 della velocità o della pressione scende sotto il minimo dichiarato
  dal caso nelle coppie asintotiche;
- non rimangono almeno due coppie valide sopra il floor di arrotondamento.

Per un ordine teorico pari a due, il valore iniziale consigliato per la soglia è
`1.8`; se una grandezza ha un ordine atteso differente, il caso deve dichiararlo
esplicitamente e documentarne la ragione. Non si accetta un test che stampi
soltanto JSON e restituisca sempre zero.

I tempi di pressione usati nel confronto esatto devono seguire lo staggering
del solver per ogni valore di `dt`; non si confrontano risultati appartenenti a
livelli temporali differenti.

### 20.6 Backend e kernel

Lo stesso sorgente `test_correctness.c` viene compilato due volte:

```text
test_correctness_standard
test_correctness_optimized
```

I due eseguibili differiscono soltanto per `SOLVER_BACKEND`. In questo modo ogni
backend deve soddisfare indipendentemente le stesse proprietà fisiche senza
introdurre dispatch runtime nel solver.

`test_kernel_equivalence` collega invece i simboli concreti dei due backend e,
su griglie piccole deterministiche, confronta l'uscita completa di ciascuno dei
sei stadi:

- momentum Dxx, Dyy e Dzz;
- pressione Dxx, Dyy e Dzz.

Il backend standard è l'oracolo implementativo; la soluzione manifatturata resta
l'oracolo fisico. Il confronto usa una tolleranza assoluta e relativa coerente
con `Real`, controlla anche le boundary e non contiene soglie temporali.

### 20.7 Integrazione nel build

CTest deve distinguere test veloci e studi costosi:

```text
correctness   test end-to-end standard e ottimizzato
kernel        equivalenza dei singoli kernel
convergence   studi spaziali e temporali
sanitizer     ownership, bounds e use-after-free sul backend standard
```

Il target di test predefinito esegue `correctness` e `kernel`. Gli studi di
convergenza sono etichettati come lenti ma devono essere eseguiti prima di
integrare cambiamenti numerici o nuove ottimizzazioni. AddressSanitizer e
UndefinedBehaviorSanitizer vengono eseguiti su griglie piccole; non si usano i
risultati di timing di build instrumentate.

I commenti nei test spiegano le formule manifatturate, la normalizzazione della
pressione e la scelta del raffinamento. Commenti come `initialize fields`,
`compute errors` o `cleanup` devono essere sostituiti da funzioni dai nomi chiari.

## 21. Strategia di migrazione

La migrazione dovrebbe avvenire per passi verificabili:

1. introdurre `ScalarField`, `VectorField` e `Direction` mantenendo i kernel
   attuali;
2. introdurre `SolverState`, senza ancora ridurre il numero di campi;
3. introdurre `SolverWorkspace` e spostare fuori dai timestep tutte le
   allocazioni;
4. creare `Solver` e ridurre `main()` a init/solve/destroy;
5. sostituire l'output asincrono con il writer sincrono;
6. introdurre `kernels.h` con gli alias compile-time e compilare inizialmente
   entrambi i backend con wrapper sui kernel esistenti;
7. implementare il backend standard memory-aware senza intrinsics;
8. adattare i tre kernel momentum ottimizzati e i tre kernel pressione indicati;
9. rimuovere `GField`, `Xi`, `Delta`, `K` e `Beta` dopo aver verificato
   l'equivalenza del percorso fuso;
10. ridurre i tre campi di pressione intermedi a un buffer di workspace più il
    riuso, dopo il momentum, dello storage di `pressure_star`;
11. elaborare il RHS del momentum una componente alla volta;
12. aggiungere i report gerarchici dei tempi e confrontare i due eseguibili;
13. solo dopo aver superato i test di convergenza, lavorare sulle nuove
    ottimizzazioni Dxx.

Ogni passo deve mantenere i test di soluzione manifatturata e di convergenza. I
confronti devono includere almeno soluzione finale, norme d'errore e ordine di
convergenza.

## 22. Decisioni

Sono state confermate le seguenti decisioni:

1. la permeabilità `K` può variare nello spazio ma non nel tempo;
2. il backend standard deve essere memory-aware e privo di SIMD esplicita;
3. il backend ottimizzato usa le versioni SIMD indicate;
4. il backend viene selezionato a compile-time;
5. le statistiche includono kernel, momentum, pressione, timestep e init;
6. i valori assoluti e gli incrementi delle boundary conditions hanno la
   semantica descritta nella sezione 4.2;
7. `pressure_star` deve persistere perché è necessario al calcolo di `g` in Dxx;
8. Dxx di velocità e pressione è la principale area di ottimizzazione futura;
9. le tre componenti della velocità hanno lo stesso numero di elementi;
10. la velocità è memorizzata ai tempi interi, mentre pressione e correzione sono
    memorizzate ai tempi seminteri;
11. al primo passo `pressure_star^(1/2)` è inizializzata con la pressione esatta
    a `t=dt/2`;
12. `g` mantiene i coefficienti `nu/K` e `nu` usati da `g_value()`;
13. il momentum mantiene tre sistemi di Thomas, incluso Dzz per `U`;
14. ogni nuova astrazione deve eliminare una complessità concreta e già
    osservabile;
15. i commenti documentano contratti, motivazioni e matematica, non istruzioni
    banali o vecchie implementazioni;
16. correttezza e convergenza usano gli stessi casi manifatturati e lo stesso
    runner, compilati per entrambi i backend.
