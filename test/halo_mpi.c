/*
 * Le celle di contorno contengono davvero i dati del vicino?
 *
 * Ogni processo riempie le celle che possiede con un numero che dipende solo
 * dalla posizione della cella nel dominio, e mette un valore riconoscibile
 * nell'anello di contorno. Dopo lo scambio, ogni cella dell'anello deve
 * contenere il numero che spetta alla sua posizione: è una copia, quindi
 * l'errore atteso è esattamente zero.
 *
 * Si controlla anche il caso opposto: dove il vicino non c'è, perché lì
 * finisce il dominio, l'anello deve essere rimasto intatto. Se qualcuno
 * scrivesse lì dei dati, il solutore userebbe valori inventati al posto delle
 * condizioni al contorno.
 */
#include <stdio.h>
#include <stdlib.h>

#include "decomp.h"
#include "parallel.h"
#include "solver.h"
#include "utils.h"

#define SEGNAPOSTO ((Real)-987654.0)

/*
 * Numero che identifica una cella dalla sua posizione nel dominio: e' il suo
 * indice globale, quindi due celle diverse hanno sempre numeri diversi.
 */
static Real tag_of(const Decomp *d, int gi, int gj, int gk) {
    long long plane = (long long)d->n_global[0] * d->n_global[1];
    return (Real)((long long)gk * plane +
                  (long long)gj * d->n_global[0] + gi);
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    int process_grid[3] = {1, 1, 0};
    if (argc == 4) {
        for (int c = 0; c < 3; c++) {
            process_grid[c] = atoi(argv[c + 1]);
        }
    }
    par_topology_init(process_grid);

    Decomp d;
    decomp_init_mpi(&d);

    Real *field = xmalloc(d.n_cells * sizeof(Real));

    /* Tutto segnaposto, comprese le celle possedute: cosi' un anello che non
     * viene riempito resta riconoscibile. */
    for (size_t i = 0; i < d.n_cells; i++) {
        field[i] = SEGNAPOSTO;
    }

    for (int k = 0; k < d.n[2]; k++) {
        for (int j = 0; j < d.n[1]; j++) {
            for (int i = 0; i < d.n[0]; i++) {
                field[decomp_index(&d, i, j, k)] =
                    tag_of(&d, decomp_global(&d, i, 0),
                           decomp_global(&d, j, 1),
                           decomp_global(&d, k, 2));
            }
        }
    }

    par_exchange_halo(&d, field);

    long long wrong_values = 0;   /* anello riempito male          */
    long long wrong_walls = 0;    /* anello scritto dove non serve */

    for (int axis = 0; axis < 3; axis++) {
        int first = (axis + 1) % 3;
        int second = (axis + 2) % 3;

        /* I due anelli: sotto la prima faccia e sopra l'ultima. */
        for (int side = 0; side < 2; side++) {
            int slot = (side == 0) ? -1 : d.n[axis];
            int step = (side == 0) ? -1 : +1;
            int neighbor = par_neighbor(axis, step);
            int cell[3];

            cell[axis] = slot;
            for (int b = 0; b < d.n[second]; b++) {
                cell[second] = b;
                for (int a = 0; a < d.n[first]; a++) {
                    cell[first] = a;

                    Real got = field[decomp_index(&d, cell[0], cell[1],
                                                  cell[2])];

                    if (neighbor == PAR_NO_NEIGHBOR) {
                        wrong_walls += (got != SEGNAPOSTO);
                    } else {
                        Real want = tag_of(&d,
                                           decomp_global(&d, cell[0], 0),
                                           decomp_global(&d, cell[1], 1),
                                           decomp_global(&d, cell[2], 2));
                        wrong_values += (got != want);
                    }
                }
            }
        }
    }

    long long bad_values = par_sum_long(wrong_values);
    long long bad_walls = par_sum_long(wrong_walls);
    int failed = (bad_values != 0) || (bad_walls != 0);

    int dims[3];
    par_dims(dims);

    if (par_rank() == 0) {
        printf("\nScambio delle celle di contorno:\n");
        printf("  processi %d x %d x %d su una griglia %d x %d x %d\n",
               dims[0], dims[1], dims[2],
               d.n_global[0], d.n_global[1], d.n_global[2]);
        printf("  celle di contorno sbagliate:       %lld\n", bad_values);
        printf("  celle scritte dove c'e' la parete: %lld\n", bad_walls);
        printf("\n  %s\n", failed
               ? "FALLITO: l'anello non contiene i dati del vicino"
               : "PASSATO: l'anello contiene i dati del vicino");
    }

    free(field);
    par_finalize();
    return failed ? 1 : 0;
}
