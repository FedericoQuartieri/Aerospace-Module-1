/*
 * Memory-layout invariance test.
 *
 * The kernels used to reach a cell with k * WIDTH * HEIGHT + j * WIDTH + i,
 * which is only correct while the owned cells are contiguous and start at
 * offset zero.  They now address memory through Decomp::stride and
 * Decomp::base instead, so the same computation must give the same answer on
 * a different layout.
 *
 * The run is performed twice on identical physics: once on the compact block,
 * once on a block allocated with a one-cell margin on every face.  In the
 * second layout consecutive rows are two cells apart, consecutive planes are
 * a full border apart, and the first owned cell no longer sits at index 0.
 * Every field must come out bit-identical.
 *
 * This is what phase C will rely on: once ghost cells carry real data, the
 * only thing that has to change is how the Decomp is filled in.
 */
#include "solver.h"
#include "field.h"

static int compare_field(const Decomp *a, const Decomp *b,
                         const Real *field_a, const Real *field_b,
                         const char *name) {
    Real worst = (Real)0;
    size_t mismatches = 0;

    for (int k = 0; k < a->n[2]; k++) {
        for (int j = 0; j < a->n[1]; j++) {
            for (int i = 0; i < a->n[0]; i++) {
                Real va = field_a[decomp_index(a, i, j, k)];
                Real vb = field_b[decomp_index(b, i, j, k)];

                if (va != vb) {
                    Real difference = (Real)fabs((double)(va - vb));
                    if (difference > worst) {
                        worst = difference;
                    }
                    mismatches++;
                }
            }
        }
    }

    printf("  %-10s mismatching cells: %zu", name, mismatches);
    if (mismatches > 0) {
        printf("   worst |difference|: %.3e", (double)worst);
    }
    printf("\n");

    return mismatches > 0;
}

static void describe(const char *label, const Decomp *d) {
    printf("  %-8s stride = {%zu, %zu, %zu}   base = %zu   allocated = %zu\n",
           label, d->stride[0], d->stride[1], d->stride[2],
           d->base, d->n_cells);
}

int main(void)
{
    Decomp compact;
    Decomp padded;

    decomp_init_serial(&compact);
    decomp_init_serial_padded(&padded, 1);

    Data compact_data = paper_data;
    Data padded_data = paper_data;
    SolverMemState compact_state;
    SolverMemState padded_state;
    SolverStats compact_stats = {0};
    SolverStats padded_stats = {0};

    solver_init(&compact, &compact_state, &compact_data, NULL);
    solver_solve(&compact, &compact_state, &compact_data, &compact_stats, 0);

    solver_init(&padded, &padded_state, &padded_data, NULL);
    solver_solve(&padded, &padded_state, &padded_data, &padded_stats, 0);

    printf("\nMemory-layout invariance:\n");
    printf("  Grid: %d x %d x %d\n",
           compact.n_global[0], compact.n_global[1], compact.n_global[2]);
    describe("compact", &compact);
    describe("padded", &padded);

    int failed = 0;
    failed |= compare_field(&compact, &padded,
                            compact_state.u.v_x, padded_state.u.v_x, "u_x");
    failed |= compare_field(&compact, &padded,
                            compact_state.u.v_y, padded_state.u.v_y, "u_y");
    failed |= compare_field(&compact, &padded,
                            compact_state.u.v_z, padded_state.u.v_z, "u_z");
    failed |= compare_field(&compact, &padded,
                            compact_state.pressure.v,
                            padded_state.pressure.v, "pressure");

    printf("\n  %s\n", failed ? "FAILED: layout changes the result"
                              : "PASSED: layout does not change the result");

    return failed ? 1 : 0;
}
