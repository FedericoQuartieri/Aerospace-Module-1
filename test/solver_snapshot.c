/* A machine-readable snapshot of owned cells, independent of MPI layout and
 * Real precision. check_pipeline.py reconstructs global fields and compares
 * all eleven state arrays to a serial scalar Schur reference. */
#include <stdint.h>
#include <string.h>
#include "params.h"
#include "test_state.h"

int main(int argc, char **argv) {
    par_init(&argc, &argv);
    if (argc != 4 && argc != 7) {
        if (!par_rank()) fprintf(stderr, "usage: %s config scenario output-prefix [px py pz]\n", argv[0]);
        par_abort(2);
    }
    params_load(argv[1]);
    int dims[3] = {0,0,0};
    if (argc == 7) for (int a = 0; a < 3; a++) dims[a] = atoi(argv[a+4]);
    par_topology_init(dims);
    Data data;
    if (!strcmp(argv[2], "paper_variable")) {
        data = paper_data;
        data.porosity_fn = test_variable_porosity;
        data.porosity_time_dependent = 1;
    } else {
        const Data *found = data_by_name(argv[2]);
        if (!found) par_abort(2);
        data = *found;
    }
    Decomp d;
    SolverMemState state;
    SolverStats stats = {0};
    decomp_init_mpi(&d);
    solver_init(&d, &state, &data, NULL);
    solver_solve(&d, &state, &data, &stats, 0);
    char path[4096];
    int length = snprintf(path, sizeof path, "%s.%d.bin", argv[3], par_rank());
    if (length < 0 || (size_t)length >= sizeof path) par_abort(2);
    FILE *out = fopen(path, "wb");
    if (!out) { perror(path); par_abort(2); }
    int32_t header[9];
    for (int a = 0; a < 3; a++) {
        header[a] = d.n_global[a]; header[3+a] = d.start[a]; header[6+a] = d.n[a];
    }
    if (fwrite(header, sizeof header, 1, out) != 1) par_abort(2);
    Real *fields[11]; test_state_fields(&state, fields);
    for (int f = 0; f < 11; f++)
        for (int k = 0; k < d.n[2]; k++)
            for (int j = 0; j < d.n[1]; j++)
                for (int i = 0; i < d.n[0]; i++) {
                    double value = fields[f][decomp_index(&d,i,j,k)];
                    if (!isfinite(value) || fwrite(&value, sizeof value, 1, out) != 1)
                        par_abort(1);
                }
    if (fclose(out)) par_abort(2);
    test_state_free(&state);
    par_finalize();
    return 0;
}
