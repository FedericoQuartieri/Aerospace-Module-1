#include "field.h"
#include "solver.h"
#include "utils.h"
#include "workers.h"

/* Allocate the array of cells for a scalar field */
void scalarField_alloc(const Decomp *d, ScalarField *sf) {
   size_t field_size = d->n_cells * sizeof(Real);
   sf->v = xmalloc(field_size);
}

/* Allocate the array of cells for each component of a vector field */
void vectorField_alloc(const Decomp *d, VectorField *vf) {
    size_t field_size = d->n_cells * sizeof(Real);
    vf->v_x = xmalloc(field_size);
    vf->v_y = xmalloc(field_size);
    vf->v_z = xmalloc(field_size);
}

/*
 * The sampled functions describe the physical problem, so they are evaluated
 * at the *global* position of each cell.  Only the loop bounds and the memory
 * offset use local indices.
 *
 * Il riempimento e' spartito fra i thread, ed e' il caso piu' facile che
 * esista: ogni cella calcola il proprio valore dalle sole coordinate e lo
 * scrive in un indice suo. Nessuna dipendenza fra iterazioni, nessuna somma da
 * riordinare, quindi il risultato resta identico bit per bit a quello di un
 * thread solo -- la stessa proprieta' che vale per i sistemi tridiagonali, e
 * qui per un motivo ancora piu' semplice.
 *
 * Non spartirlo costava caro. Quando la permeabilita' dipende dal tempo questo
 * ciclo gira a ogni passo su tutta la griglia, e restava l'unico pezzo seriale
 * mentre tutto il resto era gia' threadato: su 256^3 erano il 25% del passo a
 * un thread e il 63% a cinquantasei, cioe' un tetto di Amdahl attorno a 4x
 * qualunque cosa facessero gli altri stadi.
 */
void vectorField_fill(const Decomp *restrict d,
                      VectorField *restrict vf,
                      VectorFunction vector_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v_x = vf->v_x;
    Real *restrict v_y = vf->v_y;
    Real *restrict v_z = vf->v_z;
    const int nk = d->n[2];
    const int nj = d->n[1];
    const int ni = d->n[0];

    /* gk sta dentro il ciclo su j e non fra i due: collapse vuole due cicli
     * perfettamente annidati, e ricalcolarlo per riga non costa niente. */
    WORKERS_PARALLEL_FOR_2(workers_many())
    for (int k = 0; k < nk; k++) {
        for (int j = 0; j < nj; j++) {
            int gk = decomp_global(d, k, 2);
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < ni; i++) {
                int gi = decomp_global(d, i, 0);
                Real x = centered_physical_coord(gi, 0);
                Real y = centered_physical_coord(gj, 1);
                Real z = centered_physical_coord(gk, 2);
                size_t off = row + (size_t)i;

                v_x[off] =
                    vector_fn(staggered_physical_coord(gi, 0),
                              y, z, time, 0);
                v_y[off] =
                    vector_fn(x, staggered_physical_coord(gj, 1),
                              z, time, 1);
                v_z[off] =
                    vector_fn(x, y,
                              staggered_physical_coord(gk, 2),
                              time, 2);
            }
        }
    }
}

void scalarField_fill(const Decomp *restrict d,
                      ScalarField *restrict sf,
                      ScalarFunction scalar_fn,
                      Real t_step) {
    Real time = time_physical_coord(t_step);
    Real *restrict v = sf->v;
    const int nk = d->n[2];
    const int nj = d->n[1];
    const int ni = d->n[0];

    /* Stessa forma e stesse ragioni di vectorField_fill. */
    WORKERS_PARALLEL_FOR_2(workers_many())
    for (int k = 0; k < nk; k++) {
        for (int j = 0; j < nj; j++) {
            int gk = decomp_global(d, k, 2);
            int gj = decomp_global(d, j, 1);
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < ni; i++) {
                int gi = decomp_global(d, i, 0);
                Real x = centered_physical_coord(gi, 0);
                Real y = centered_physical_coord(gj, 1);
                Real z = centered_physical_coord(gk, 2);

                v[row + (size_t)i] = scalar_fn(x, y, z, time);
            }
        }
    }
}
