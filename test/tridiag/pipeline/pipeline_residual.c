/* Check the equations themselves, including MPI junctions and one-cell
 * blocks. This does not depend on Schur supporting the same decomposition. */
#include <string.h>
#include "backend.h"
#include "backend_pipeline.h"
#include "momentum_row.h"
#include "pressure_common.h"
#include "params.h"
#include "test_state.h"

static Real residual(Real a, Real b, Real c, Real f, Real left, Real x, Real right) {
    Real ax = a*left, bx = b*x, cx = c*right;
    Real scale = (Real)1 + fabs(ax) + fabs(bx) + fabs(cx) + fabs(f);
    Real r = fabs(ax + bx + cx - f) / scale;
    return isfinite(r) ? r : (Real)INFINITY;
}

int main(int argc, char **argv) {
    par_init(&argc, &argv);
    if (argc != 2 && argc != 5) par_abort(2);
    params_load(argv[1]);
    int dims[3] = {0,0,0};
    if (argc == 5) for (int a = 0; a < 3; a++) dims[a] = atoi(argv[a+2]);
    par_topology_init(dims);
    Decomp d;
    SolverMemState state;
    Data data = paper_data;
    data.porosity_fn = test_variable_porosity;
    decomp_init_mpi(&d);
    solver_init(&d, &state, &data, NULL);
    backend_init(&d, &state);
    Real *fields[11]; test_state_fields(&state, fields);
    Real worst = 0;
    size_t bytes = d.n_cells * sizeof(Real);
    Real *old[3], *increment[3];
    MomentumRow *rows[3];
    for (int c = 0; c < 3; c++) {
        old[c] = xmalloc(bytes); increment[c] = xmalloc(bytes);
        rows[c] = xmalloc(d.n_cells * sizeof(MomentumRow));
    }
    for (int axis = 0; axis < 3; axis++) {
        for (int f = 0; f < 11; f++) par_exchange_halo(&d, fields[f]);
        for (int c = 0; c < 3; c++) {
            memcpy(old[c], fields[3*axis+c], bytes);
            MomentumLine ctx = momentum_line(&d, &state, &data, 1, c, axis);
            for (int k = 0; k < d.n[2]; k++)
                for (int j = 0; j < d.n[1]; j++)
                    for (int i = 0; i < d.n[0]; i++) {
                        int cell[3] = {i,j,k}; size_t at = decomp_index(&d,i,j,k);
                        rows[c][at] = momentum_row(&ctx,cell,at);
                    }
        }
        pipeline_momentum_direction(&d, &state, &data, axis, 1);
        for (int c = 0; c < 3; c++) {
            for (int k = 0; k < d.n[2]; k++)
                for (int j = 0; j < d.n[1]; j++)
                    for (int i = 0; i < d.n[0]; i++) {
                        size_t at = decomp_index(&d,i,j,k);
                        increment[c][at] = fields[3*axis+c][at] - old[c][at];
                    }
            par_exchange_halo(&d, increment[c]);
            for (int k = 0; k < d.n[2]; k++)
                for (int j = 0; j < d.n[1]; j++)
                    for (int i = 0; i < d.n[0]; i++) {
                        size_t at = decomp_index(&d,i,j,k), step = d.stride[axis];
                        MomentumRow row = rows[c][at];
                        Real r = residual(row.a,row.b,row.c,row.f,
                            row.a != 0 ? increment[c][at-step] : 0,
                            increment[c][at],row.c != 0 ? increment[c][at+step] : 0);
                        if (r > worst) worst = r;
                    }
        }
    }
    for (int axis = 0; axis < 3; axis++) {
        Real *a = xmalloc((size_t)d.n[axis] * sizeof(Real));
        Real *b = xmalloc((size_t)d.n[axis] * sizeof(Real));
        Real *c = xmalloc((size_t)d.n[axis] * sizeof(Real));
        pressure_matrix(&d,axis,a,b,c);
        for (int repeat = 0; repeat < 2; repeat++) {
            for (int k = 0; k < d.n[2]; k++)
                for (int j = 0; j < d.n[1]; j++)
                    for (int i = 0; i < d.n[0]; i++)
                        old[0][decomp_index(&d,i,j,k)] =
                            sin((i+d.start[0])*.3 + (j+d.start[1])*.5 + (k+d.start[2])*.7 + repeat);
            pipeline_pressure_direction(&d,state.backend,axis,old[0],increment[0]);
            par_exchange_halo(&d,increment[0]);
            for (int k = 0; k < d.n[2]; k++)
                for (int j = 0; j < d.n[1]; j++)
                    for (int i = 0; i < d.n[0]; i++) {
                        int cell[3] = {i,j,k}, level = cell[axis];
                        size_t at = decomp_index(&d,i,j,k), step = d.stride[axis];
                        Real r = residual(a[level],b[level],c[level],old[0][at],
                            a[level] != 0 ? increment[0][at-step] : 0,
                            increment[0][at],c[level] != 0 ? increment[0][at+step] : 0);
                        if (r > worst) worst = r;
                    }
        }
        free(a); free(b); free(c);
    }
    worst = par_max_real(worst);
#ifdef USE_FLOAT
    Real tolerance = (Real)2e-5;
#else
    Real tolerance = (Real)1e-11;
#endif
    int failed = !(worst <= tolerance);
    if (!par_rank()) printf("pipeline scaled residual: %.17g %s\n", (double)worst, failed ? "FAILED" : "PASSED");
    for (int c = 0; c < 3; c++) { free(old[c]); free(increment[c]); free(rows[c]); }
    backend_free(&state); test_state_free(&state);
    par_finalize();
    return failed;
}
