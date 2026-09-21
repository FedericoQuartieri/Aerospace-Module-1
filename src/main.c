#include <stdio.h>
#include "solver.h"
#include "parallel.h"

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s [scenario]\n"
            "       %s [config-file] [scenario]\n"
            "\nWith no arguments it uses the default values and the "
            "`paper_data` scenario.\nWith a single argument, if the name is a "
            "scenario it runs it; otherwise it is read as a config file.\n"
            "Available scenarios:\n", program, program);
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

    const char *config_path = NULL;
    const char *data_name = NULL;

    if (argc == 2) {
        if (data_by_name(argv[1]) != NULL) {
            data_name = argv[1];
        } else {
            config_path = argv[1];
        }
    } else if (argc == 3) {
        config_path = argv[1];
        data_name = argv[2];
    }

    /*
     * The configuration must be read before decomp_init_mpi, which is the
     * first to ask how large the grid is. Every process reads it: it is a file
     * of a few lines, so nobody has to send the parameters to the others.
     */
    if (config_path != NULL) {
        params_load(config_path);
    }

    SolverMemState solver_mem_state;
    Data data = paper_data;
    SolverStats solver_stats = {0};
    Decomp decomp;

    /* All zeros: MPI chooses the shape of the process grid. */
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
