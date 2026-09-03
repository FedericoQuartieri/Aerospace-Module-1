/*
 * Il binario delle misure: paper_data, griglia da file, forma della griglia
 * di processi imposta dalla riga di comando.
 *
 * Esiste perche' nessuno dei due binari gia' presenti sa fare entrambe le
 * cose, e uno studio di scaling ha bisogno di tutte e due:
 *
 *   src/main.c      legge il file di configurazione (griglia a runtime, un
 *                   binario per tutte le taglie) ma lascia scegliere la forma
 *                   a MPI_Dims_create, che non si puo' contraddire.
 *   test/paper_man  accetta la forma sulla riga di comando ma ha la griglia
 *                   cablata a compilazione, e ricalcola la permeabilita' a
 *                   ogni passo con i cicli seriali di field.c -- lavoro che
 *                   non scala e che coprirebbe l'effetto cercato.
 *
 * Qui la griglia si legge da file e la forma si impone, cosi' "quale asse
 * viene diviso" diventa una variabile dello studio invece di una conseguenza.
 * Lo scenario e' `paper_data` (K statico), lo stesso delle misure di
 * MULTITHREAD.md §9.
 *
 * Stampa, oltre alle statistiche di print_stats, le righe che servono a
 * riconoscere la configurazione nel log senza fidarsi di come e' stato
 * lanciato: backend compilato, batch della pipeline, forma effettiva della
 * griglia di processi, memoria di picco. Sono l'unico modo di accorgersi che
 * si stanno confrontando due cose diverse credendole uguali.
 *
 *   uso: bench <config> [px py pz]
 *
 * Con BENCH_NORMS=1 nell'ambiente calcola anche le norme dell'errore contro
 * la soluzione esatta: la stessa configurazione che si sta cronometrando dice
 * pure se sta risolvendo il problema giusto, che e' la premessa di qualunque
 * tempo misurato.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#include "solver.h"
#include "params.h"
#include "parallel.h"
#include "error_norms.h"

static const char *backend_name(void)
{
#if defined(TRIDIAG_PIPELINE)
    return "pipeline";
#elif defined(TRIDIAG_SCHUR)
    return "schur";
#else
    return "sconosciuto";
#endif
}

/* Zero quando il backend non ha questo parametro: la colonna resta, il valore
 * dice che non si applica. */
static int backend_batch_lines(void)
{
#if defined(PIPELINE_BATCH_LINES)
    return PIPELINE_BATCH_LINES;
#else
    return 0;
#endif
}

static int built_with_simd(void)
{
#if defined(USE_SIMD)
    return 1;
#else
    return 0;
#endif
}

static int built_with_omp(void)
{
#if defined(USE_OMP)
    return 1;
#else
    return 0;
#endif
}

static int built_with_mpi(void)
{
#if defined(USE_MPI)
    return 1;
#else
    return 0;
#endif
}

/*
 * Il massimo fra i processi, non quello del rank 0: la pipeline tiene c' e d'
 * di tutto il blocco locale, e con una decomposizione sbilanciata il blocco
 * piu' grande e' quello che decide se il caso ci sta in memoria.
 */
static long long peak_rss_kb(void)
{
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return (long long)usage.ru_maxrss;
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    if (argc != 2 && argc != 5) {
        if (par_rank() == 0) {
            fprintf(stderr,
                    "uso: %s <file-di-configurazione> [px py pz]\n"
                    "\nSenza px py pz la forma la sceglie MPI_Dims_create.\n"
                    "BENCH_NORMS=1 aggiunge le norme dell'errore.\n",
                    argv[0]);
        }
        par_finalize();
        return 1;
    }

    params_load(argv[1]);

    /* Tutti zero: la forma la sceglie MPI. Tre numeri la impongono, ed e' il
     * motivo per cui questo binario esiste. */
    int process_grid[3] = {0, 0, 0};

    if (argc == 5) {
        for (int c = 0; c < 3; c++) {
            process_grid[c] = atoi(argv[c + 2]);
        }
    }

    par_topology_init(process_grid);

    Decomp decomp;
    SolverMemState solver_mem_state;
    SolverStats solver_stats = {0};
    Data data = paper_data;

    decomp_init_mpi(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, NULL);

    const int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    /* Prima delle norme: calcolarle alloca due campi in piu' e falserebbe la
     * memoria di picco del caso che si sta misurando. */
    const long long rss_kb = par_max_long(peak_rss_kb());

    int dims[3];
    par_dims(dims);

    if (par_rank() == 0) {
        printf("  bench backend:     %s\n", backend_name());
        printf("  bench batch:       %d\n", backend_batch_lines());
        printf("  bench build:       simd=%d omp=%d mpi=%d\n",
               built_with_simd(), built_with_omp(), built_with_mpi());
        printf("  bench ranks:       %d\n", par_size());
        printf("  bench proc grid:   %d x %d x %d\n",
               dims[0], dims[1], dims[2]);
        printf("  bench global grid: %d x %d x %d\n",
               decomp.n_global[0], decomp.n_global[1], decomp.n_global[2]);
        printf("  bench peak rss:    %.1f MB\n", (double)rss_kb / 1024.0);
        fflush(stdout);
    }

    /* Il valore, non la sola presenza: gli script esportano sempre la
     * variabile (mpirun -x vuole che esista) e la mettono a 0 quando le norme
     * non servono. */
    const char *norms_requested = getenv("BENCH_NORMS");

    if (norms_requested != NULL && norms_requested[0] != '\0' &&
        norms_requested[0] != '0') {
        const Real velocity_verification_time = (Real)STEPS * (Real)DT;
        const Real pressure_verification_time =
            velocity_verification_time - (Real)DT / 2.0;
        const SolverErrorNorms errors =
            compute_solver_error_norms(&decomp, &solver_mem_state, &data,
                                       velocity_verification_time,
                                       pressure_verification_time);

        print_solver_error_norms(&decomp, &errors,
                                 velocity_verification_time,
                                 pressure_verification_time);
    }

    par_finalize();
    return 0;
}
