/*
 * The measurement binary: paper_data, grid from file, shape of the process
 * grid imposed by the command line.
 *
 * It exists because neither of the two binaries already present can do both
 * things, and a scaling study needs both:
 *
 *   src/main.c      reads the configuration file (grid at run time, one
 *                   binary for all the sizes) but leaves the choice of shape
 *                   to MPI_Dims_create, which cannot be contradicted.
 *   test/paper_man  accepts the shape on the command line but has the grid
 *                   wired in at compile time, and recomputes the permeability
 *                   at every step with the serial loops of field.c -- work
 *                   that does not scale and that would mask the effect being
 *                   sought.
 *
 * Here the grid is read from a file and the shape is imposed, so "which axis
 * is divided" becomes a variable of the study instead of a consequence. The
 * scenario is `paper_data` (static K), the same as the measurements of
 * MULTITHREAD.md §9.
 *
 * Besides the statistics of print_stats, it prints the lines needed to
 * recognise the configuration in the log without trusting how it was launched:
 * compiled backend, pipeline batch, actual shape of the process grid, peak
 * memory. They are the only way to notice that two different things are being
 * compared believing them equal.
 *
 *   usage: bench <config> [px py pz]
 *
 * With BENCH_NORMS=1 in the environment it also computes the error norms
 * against the exact solution: the same configuration that is being timed also
 * says whether it is solving the right problem, which is the premise of any
 * measured time.
 *
 * BENCH_SCENARIO=<name> changes the scenario; without it, it is `paper_data'.
 * It is needed because not all of them cost the same: the forcing term is
 * supplied by the scenario, and paper_data is the only one that publishes its
 * line version (forcing_line_fn). Measuring only it means measuring only the
 * best case, and a change to the right-hand side would seem to be worth for
 * all of them as much as it is for one. The names are those of
 * data_print_names.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#include "solver.h"
#include "params.h"
#include "parallel.h"
#include "error_norms.h"

#include "backend.h"

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
 * The maximum among the processes, not that of rank 0: the pipeline keeps c'
 * and d' of the whole local block, and with an unbalanced decomposition the
 * largest block is the one that decides whether the case fits in memory.
 */
static long long peak_rss_kb(void)
{
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#ifdef __APPLE__
    return (long long)usage.ru_maxrss / 1024;
#else
    return (long long)usage.ru_maxrss;
#endif
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    if (argc != 2 && argc != 5) {
        if (par_rank() == 0) {
            fprintf(stderr,
                    "usage: %s <config-file> [px py pz]\n"
                    "\nWithout px py pz the shape is chosen by "
                    "MPI_Dims_create.\n"
                    "BENCH_NORMS=1 adds the error norms.\n",
                    argv[0]);
        }
        par_finalize();
        return 1;
    }

    params_load(argv[1]);

    /* The scenario: the default one, or the one requested by the environment.
     * A wrong name stops here instead of silently measuring something else. */
    const char *scenario = getenv("BENCH_SCENARIO");
    const Data *scelto = NULL;

    if (scenario != NULL && scenario[0] != '\0') {
        scelto = data_by_name(scenario);
        if (scelto == NULL) {
            if (par_rank() == 0) {
                fprintf(stderr, "unknown scenario: %s\nAvailable:\n",
                        scenario);
                data_print_names(stderr);
            }
            par_abort(1);
        }
    }

    /* All zeros: MPI chooses the shape. Three numbers impose it, and that is
     * the reason this binary exists. */
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
    Data data = (scelto != NULL) ? *scelto : paper_data;

    decomp_init_mpi(&decomp);
    solver_init(&decomp, &solver_mem_state, &data, NULL);

    const int write_enabled = 0;
    solver_solve(&decomp, &solver_mem_state, &data, &solver_stats,
                 write_enabled);

    /* Before the norms: computing them allocates two more fields and would
     * falsify the peak memory of the case being measured. */
    const long long rss_kb = par_max_long(peak_rss_kb());

    int dims[3];
    par_dims(dims);

    if (par_rank() == 0) {
        printf("  bench backend:     %s\n", backend_name());
        /* The scenario in the output, not only in the environment: a CSV that
         * does not report it does not say which problem it timed. */
        printf("  bench scenario:    %s\n", data.name);
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

    /* The value, not just the presence: the scripts always export the variable
     * (mpirun -x wants it to exist) and set it to 0 when the norms are not
     * needed. */
    int failed = 0;
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
        failed = !isfinite(errors.velocity_x.L2) || !isfinite(errors.velocity_y.L2) ||
                 !isfinite(errors.velocity_z.L2) || !isfinite(errors.pressure.L2);
    }

    par_finalize();
    return failed;
}
