#include <stdio.h>
#include <string.h>
#include "solver.h"
#include "parallel.h"

static void usage(const char *program) {
    fprintf(stderr,
            "uso: %s [file-di-configurazione] [scenario]\n"
            "\nSenza argomenti usa i valori di default e lo scenario "
            "`paper_data`.\nScenari disponibili:\n", program);
    data_print_names(stderr);
}

int main(int argc, char **argv) {
    par_init(&argc, &argv);

    if (argc > 3) {
        if (par_rank() == 0) {
            usage(argv[0]);
        }
        par_finalize();
        return 1;
    }

    /*
     * La configurazione va letta prima di decomp_init_mpi, che e' il primo a
     * chiedere quanto e' grande la griglia.  La leggono tutti i processi: e' un
     * file di poche righe, e cosi' nessuno deve spedire i parametri agli altri.
     */
    if (argc >= 2) {
        params_load(argv[1]);
    }

    const char *data_name = (argc == 3) ? argv[2] : NULL;

    SolverMemState solver_mem_state;
    Data data = paper_data;
    SolverStats solver_stats = {0};
    Decomp decomp;

    /* Tutti zero: la forma della griglia di processi la sceglie MPI. */
    const int process_grid[3] = {0, 0, 0};
    par_topology_init(process_grid);
    decomp_init_mpi(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, data_name);

    int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    par_finalize();
    return 0;
}
