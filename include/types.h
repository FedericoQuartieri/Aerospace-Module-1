#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#ifdef USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

/* Function of space, time and vector component (0: X, 1: Y, 2: Z). */
typedef Real (*VectorFunction)(Real x, Real y, Real z, Real t, int component);

/* Function of space and time */
typedef Real (*ScalarFunction)(Real x, Real y, Real z, Real t);

/*
 * Il termine forzante di una linea intera lungo x, in un colpo solo.
 *
 * Il core la chiama una volta per linea invece di una volta per cella.  La
 * chiamata indiretta si ammortizza su `n` celle, ma soprattutto sparisce dal
 * ciclo interno: finche' c'e', il compilatore non puo' ne' inlinare ne'
 * vettorizzare, ed e' il motivo per cui il passo eta e' l'unico dei tre senza
 * kernel SIMD.
 *
 * Chi la implementa vede la linea intera, quindi tutto cio' che dipende solo
 * da y, z e t lo calcola una volta prima del ciclo.  Li' sta il guadagno
 * vero: per una forzante separabile come quella del paper, delle cinque
 * chiamate trigonometriche per cella ne resta una.
 *
 * Le linee corrono lungo x perche' g entra solo nel passo eta, che e' quello
 * lungo x.  Se un giorno entrasse anche negli altri, la firma andrebbe
 * generalizzata con un asse.
 *
 * `out` ha `n` elementi e la cella i sta in (xs[i], y, z).  Le ascisse
 * arrivano gia' calcolate e gia' sfalsate per la componente richiesta: le
 * prepara il core con le stesse operazioni del percorso scalare, cosi' chi
 * implementa questa funzione non puo' valutarla nel punto sbagliato ne'
 * cambiarne l'ultimo bit ricostruendo le coordinate per conto suo.
 *
 * Puo' restare NULL.  In quel caso il core ripiega su forcing_fn cella per
 * cella e lo scenario si comporta esattamente come prima, quindi gli scenari
 * che non hanno niente da guadagnare non vanno toccati.
 */
typedef void (*VectorLineFunction)(Real *restrict out, const Real *restrict xs,
                                   int n, Real y, Real z, Real t,
                                   int component);

typedef struct Data {
    const char *name;
    VectorFunction bc_velocity;
    VectorFunction forcing_fn;
    /* Versione a linea della stessa forzante, o NULL: vedi
     * VectorLineFunction. */
    VectorLineFunction forcing_line_fn;
    VectorFunction porosity_fn;
    int porosity_time_dependent;
    VectorFunction velocity_fn;
    ScalarFunction pressure_fn;
} Data;

typedef struct ScalarField {
    Real *v;
} ScalarField;

typedef struct VectorField {
    Real *v_x;
    Real *v_y;
    Real *v_z;
} VectorField;

typedef struct SolverStats {
    /* Accumulated execution times, in nanoseconds. */
    uint64_t eta_sys;
    uint64_t zeta_sys;
    uint64_t u_sys;
    uint64_t psi_sys;
    uint64_t phi_low_sys;
    uint64_t phi_high_sys;
    uint64_t pressure_update;
    /*
     * Il riempimento della permeabilita', quando dipende dal tempo.
     *
     * Non era cronometrato, e per questo era invisibile: la somma degli stadi
     * non faceva il totale e nessuno lo controllava. Su 256^3 mancava un
     * quarto del passo a un thread e quasi due terzi a cinquantasei, ed era
     * quello a fissare il tetto dello speedup mentre lo si cercava altrove.
     */
    uint64_t porosity_fill;
    uint64_t solve_steps;
    uint64_t wr_output;
} SolverStats;

#endif
