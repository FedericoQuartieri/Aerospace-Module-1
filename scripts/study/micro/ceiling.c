/*
 * I due tetti della macchina, misurati prima di giudicare qualunque scaling.
 *
 * Uno speedup si legge contro il massimo ottenibile, non contro il numero di
 * core: su un portatile a 4 core il massimo misurato era 2.91x, e un 2.18x
 * letto contro 4 sembra un fallimento mentre contro 2.91 e' il 75% del
 * possibile (MULTITHREAD.md §7.1).  I tetti sono due perche' i pezzi del
 * solutore ne toccano due diversi:
 *
 *   trig    solo calcolo, nessun accesso alla memoria.  E' il tetto della
 *           forzante e della soluzione manufatturata, che sono trigonometria
 *           pura e valgono meta' del passo su paper_data.
 *
 *   triad   a[i] = b[i] + s * c[i] su array molto piu' grandi dell'ultimo
 *           livello di cache.  E' il tetto vero di uno stencil: la banda di
 *           memoria, che i thread saturano molto prima dei core.
 *
 * Il primo tocco degli array avviene con la stessa spartizione del calcolo:
 * su due socket le pagine finiscono cosi' vicino al thread che le usera', e
 * si misura la banda della macchina invece della latenza di attraversare il
 * collegamento fra i socket.
 *
 *   uso: ceiling <trig|triad> [milioni-di-iterazioni | MB-per-array]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_seconds(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static int thread_count(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Il risultato si stampa: senza, il compilatore ha il diritto di cancellare
 * tutto il ciclo, e si misurerebbe la velocita' di non fare niente. */
static double trig_ceiling(long iterations, double *checksum)
{
    double total = 0.0;
    double start = now_seconds();

#pragma omp parallel for schedule(static) reduction(+:total)
    for (long i = 0; i < iterations; i++) {
        double x = (double)i * 1e-6;
        total += sin(x) * cos(x + 1.0) * sin(x + 2.0);
    }

    *checksum = total;
    return now_seconds() - start;
}

static double triad_ceiling(size_t count, double *checksum)
{
    double *a = malloc(count * sizeof *a);
    double *b = malloc(count * sizeof *b);
    double *c = malloc(count * sizeof *c);
    const double scalar = 3.0;
    double elapsed;

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "ceiling: memoria insufficiente\n");
        exit(1);
    }

    /* Primo tocco con la stessa spartizione del ciclo misurato. */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        a[i] = 0.0;
        b[i] = 1.0 + (double)i;
        c[i] = 2.0;
    }

    /* Un giro a vuoto: la prima passata paga ancora i page fault. */
    for (int pass = 0; pass < 2; pass++) {
        elapsed = now_seconds();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i++) {
            a[i] = b[i] + scalar * c[i];
        }
        elapsed = now_seconds() - elapsed;
    }

    *checksum = a[count / 2];
    free(c);
    free(b);
    free(a);
    return elapsed;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "trig";
    double checksum = 0.0;

    if (strcmp(mode, "trig") == 0) {
        long millions = (argc > 2) ? atol(argv[2]) : 200;
        long iterations = millions * 1000000L;
        double seconds = trig_ceiling(iterations, &checksum);

        printf("ceiling mode=trig threads=%d time_ms=%.3f "
               "rate=%.3f unit=Mop/s checksum=%.6e\n",
               thread_count(), seconds * 1e3,
               (double)iterations / seconds / 1e6, checksum);
    } else if (strcmp(mode, "triad") == 0) {
        long megabytes = (argc > 2) ? atol(argv[2]) : 512;
        size_t count = (size_t)megabytes * 1024 * 1024 / sizeof(double);
        double seconds = triad_ceiling(count, &checksum);
        /* Tre array toccati per iterazione: due letti, uno scritto. */
        double gigabytes = 3.0 * (double)count * sizeof(double) / 1e9;

        printf("ceiling mode=triad threads=%d time_ms=%.3f "
               "rate=%.3f unit=GB/s checksum=%.6e\n",
               thread_count(), seconds * 1e3, gigabytes / seconds, checksum);
    } else {
        fprintf(stderr, "uso: %s <trig|triad> [taglia]\n", argv[0]);
        return 1;
    }

    return 0;
}
