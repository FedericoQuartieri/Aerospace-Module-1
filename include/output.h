#ifndef OUTPUT_H
#define OUTPUT_H

#include "solver.h"

void write_vti_ascii(const Decomp *d,
                     const SolverMemState *solver_mem_state,
                     const char *output_directory,
                     int t_step);
void write_vti_binary(const Decomp *d,
                      const SolverMemState *solver_mem_state,
                      const char *output_directory,
                      int t_step);
void write_to_file(const Decomp *d,
                   const SolverMemState *solver_mem_state,
                   const char *data_name,
                   int t_step);

#endif
