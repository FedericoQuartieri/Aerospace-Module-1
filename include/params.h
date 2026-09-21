#ifndef PARAMS_H
#define PARAMS_H

#include "types.h"

/*
 * The simulation parameters.
 *
 * They used to be compile-time constants, and changing the grid or the time
 * step meant recompiling: that is what the convergence and scaling scripts
 * did, one binary per case. Now they live here.
 *
 * `sim` starts with the default values, which are still the ones chosen at
 * compile time (DEFAULT_WIDTH and friends in solver.h): a program that reads
 * no file behaves exactly as before.
 *
 * It is written once at start-up and only read afterwards. It must be filled
 * in before decomp_init_*, which is the first to ask how large the grid is.
 *
 * The fields after the blank line are derived from the others and are not read
 * from the file: params_load recomputes them after every read.
 */
typedef struct SimParams {
    int width, height, depth;
    Real lx, ly, lz;
    Real t_end;
    int steps;
    Real nu;
    int wr_freq;
    /*
     * Lines per pipeline batch. 0, the default, lets the pipeline choose at
     * start-up based on the threads per process; Schur ignores the key.
     */
    int pipeline_batch_lines;

    Real dx, dy, dz;
    Real dx_inverse, dy_inverse, dz_inverse;
    Real dx_inverse_square, dy_inverse_square, dz_inverse_square;
    Real dt;
} SimParams;

extern SimParams sim;

/*
 * Reads a file of `key = value` lines, one per line; blank lines and those
 * starting with '#' are comments. The keys are the names of the fields above,
 * excluding the derived ones. An unknown key or an unintelligible line stops
 * the program: a misspelt parameter is a wrong result, not a detail.
 */
void params_load(const char *path);

#endif
