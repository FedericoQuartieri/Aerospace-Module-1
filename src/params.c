#include "params.h"
#include "solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Le stesse formule di prima, quando erano macro: la spaziatura da lunghezza e
 * numero di punti, gli inversi presi una volta per non dividere nei cicli, e il
 * passo temporale da durata e numero di passi.
 */
#define SPACING(length, points) ((2 * (length)) / (Real)(2 * (points) - 1))

/*
 * I default valgono per un programma che non legge nessun file, quindi devono
 * comprendere anche i derivati: qui sono espressioni costanti, le calcola il
 * compilatore.
 */
SimParams sim = {
    .width = DEFAULT_WIDTH,
    .height = DEFAULT_HEIGHT,
    .depth = DEFAULT_DEPTH,
    .lx = DEFAULT_LX,
    .ly = DEFAULT_LY,
    .lz = DEFAULT_LZ,
    .t_end = DEFAULT_T,
    .steps = DEFAULT_STEPS,
    .nu = DEFAULT_NU,
    .wr_freq = DEFAULT_WR_FREQ,

    .dx = SPACING(DEFAULT_LX, DEFAULT_WIDTH),
    .dy = SPACING(DEFAULT_LY, DEFAULT_HEIGHT),
    .dz = SPACING(DEFAULT_LZ, DEFAULT_DEPTH),
    .dx_inverse = 1.0 / SPACING(DEFAULT_LX, DEFAULT_WIDTH),
    .dy_inverse = 1.0 / SPACING(DEFAULT_LY, DEFAULT_HEIGHT),
    .dz_inverse = 1.0 / SPACING(DEFAULT_LZ, DEFAULT_DEPTH),
    .dx_inverse_square = (1.0 / SPACING(DEFAULT_LX, DEFAULT_WIDTH)) *
                         (1.0 / SPACING(DEFAULT_LX, DEFAULT_WIDTH)),
    .dy_inverse_square = (1.0 / SPACING(DEFAULT_LY, DEFAULT_HEIGHT)) *
                         (1.0 / SPACING(DEFAULT_LY, DEFAULT_HEIGHT)),
    .dz_inverse_square = (1.0 / SPACING(DEFAULT_LZ, DEFAULT_DEPTH)) *
                         (1.0 / SPACING(DEFAULT_LZ, DEFAULT_DEPTH)),
    .dt = (Real)DEFAULT_T / (Real)DEFAULT_STEPS,
};

/* Dove mettere il valore di ogni chiave, e se e' un intero o un reale. */
static const struct {
    const char *name;
    int is_integer;
    void *slot;
} settable[] = {
    { "width",   1, &sim.width },
    { "height",  1, &sim.height },
    { "depth",   1, &sim.depth },
    { "lx",      0, &sim.lx },
    { "ly",      0, &sim.ly },
    { "lz",      0, &sim.lz },
    { "t_end",   0, &sim.t_end },
    { "steps",   1, &sim.steps },
    { "nu",      0, &sim.nu },
    { "wr_freq", 1, &sim.wr_freq },
};

static void params_derive(void) {
    sim.dx = SPACING(sim.lx, sim.width);
    sim.dy = SPACING(sim.ly, sim.height);
    sim.dz = SPACING(sim.lz, sim.depth);

    sim.dx_inverse = 1.0 / sim.dx;
    sim.dy_inverse = 1.0 / sim.dy;
    sim.dz_inverse = 1.0 / sim.dz;

    sim.dx_inverse_square = sim.dx_inverse * sim.dx_inverse;
    sim.dy_inverse_square = sim.dy_inverse * sim.dy_inverse;
    sim.dz_inverse_square = sim.dz_inverse * sim.dz_inverse;

    sim.dt = (Real)sim.t_end / (Real)sim.steps;
}

/* Un valore fuori dai valori ammessi si vede subito, non dieci passi dopo. */
static void params_check(const char *path) {
    if (sim.width < 2 || sim.height < 2 || sim.depth < 2) {
        fprintf(stderr, "%s: la griglia vuole almeno 2 punti per direzione\n",
                path);
        exit(1);
    }
    if (sim.steps < 1) {
        fprintf(stderr, "%s: steps deve essere almeno 1\n", path);
        exit(1);
    }
    if (sim.lx <= 0 || sim.ly <= 0 || sim.lz <= 0 ||
        sim.t_end <= 0 || sim.nu <= 0) {
        fprintf(stderr, "%s: lx, ly, lz, t_end e nu devono essere positivi\n",
                path);
        exit(1);
    }
}

void params_load(const char *path) {
    FILE *fp = fopen(path, "r");

    if (fp == NULL) {
        perror(path);
        exit(1);
    }

    char line[256];
    int number = 0;

    while (fgets(line, sizeof line, fp) != NULL) {
        char key[64];
        double value;

        number++;

        /* Riga vuota o commento. */
        char first = 0;
        if (sscanf(line, " %c", &first) != 1 || first == '#') {
            continue;
        }

        if (sscanf(line, " %63[a-z_] = %lf", key, &value) != 2) {
            fprintf(stderr, "%s:%d: non e' una riga `chiave = valore`: %s",
                    path, number, line);
            exit(1);
        }

        size_t count = sizeof settable / sizeof settable[0];
        size_t which;

        for (which = 0; which < count; which++) {
            if (strcmp(key, settable[which].name) == 0) {
                break;
            }
        }

        if (which == count) {
            fprintf(stderr, "%s:%d: chiave sconosciuta `%s`\n",
                    path, number, key);
            exit(1);
        }

        if (settable[which].is_integer) {
            *(int *)settable[which].slot = (int)value;
        } else {
            *(Real *)settable[which].slot = (Real)value;
        }
    }

    fclose(fp);
    params_derive();
    params_check(path);
}
