/*
 * Brinkman flow in a plane channel, against its exact solution.
 *
 * The manufactured tests keep K between 0.55 and 1, where the drag NU/K is no
 * larger than the diffusion.  The physical scenarios use K = 1e-4 inside an
 * obstacle and 1e30 around it, a regime no other test checks.  This one does,
 * with a flow whose solution is known in closed form.
 *
 * The walls are y = 0 and y = LY, and a uniform body force G pushes along X.
 * It is the same flow as one driven by a pressure gradient -G: with u
 * independent of x the two differ by a pressure that is exactly zero.  The
 * flow is steady and unidirectional, u = (U(y), 0, 0), with
 *
 *   -NU U'' + (NU / K(y)) U = G,    U(0) = U(LY) = 0.
 *
 * Two permeabilities:
 *
 *   uniform  K everywhere.  U is flat in the middle and falls to zero in two
 *            boundary layers of thickness sqrt(K) along the walls.
 *   layer    K in a porous layer 0 <= y <= LAYER_FRACTION * LY, free fluid
 *            above.  The drag jumps from NU/K to zero at the interface, as
 *            it does on the surface of an obstacle.
 *
 * The run starts from the exact solution.  The scheme advances increments,
 * so once the flow has settled the splitting terms vanish and what is left
 * is the spatial error: halving h at fixed K gives the order in space, and
 * lowering K at fixed h shows when sqrt(K) stops being resolved.
 *
 * The X and Z faces receive the exact profile, so the cells on them are
 * exact by construction.  The error is measured on the other cells only, and
 * the Makefile makes the channel wide (LX = LZ = 20 with a handful of cells)
 * so that those faces barely reach the cells that are measured.
 *
 *   brinkman_channel [uniform|layer] [K] [config-file]
 *
 * Without a configuration file the grid is the one the Makefile compiles in.
 * The run fails when the relative L2 error of the velocity exceeds 5%.
 */
#include <stdlib.h>
#include <string.h>

#include "solver.h"
#include "parallel.h"
#include "params.h"
#include "physics.h"
#include "error_norms.h"

#define LAYER_FRACTION 0.25
#define FREE_FLUID_PERMEABILITY 1e30
#define MAX_RELATIVE_ERROR 5e-2

static int layer_case = 0;
static double permeability = 1e-2;
static double body_force = 0.0;

/*
 * U(y) for uniform K, divided by G K / NU:
 *
 *   1 - cosh(a) / cosh(b),   a = |y - LY/2| / sqrt(K),   b = LY / (2 sqrt(K))
 *
 * rewritten as expm1(a - b) expm1(-a - b) / (1 + exp(-2b)), which neither
 * overflows when K is small nor cancels to zero when K is large.
 */
static double uniform_shape(double y)
{
    const double s = sqrt(permeability);
    const double a = fabs(y - 0.5 * (double)LY) / s;
    const double b = 0.5 * (double)LY / s;

    return expm1(a - b) * expm1(-a - b) / (1.0 + exp(-2.0 * b));
}

/*
 * The layer case, with D the layer thickness, L = LY - D the free gap,
 * s = sqrt(K), c = G K / NU and q = exp(-D / s):
 *
 *   porous  U = c + E exp(-(D - y) / s) + F exp(-y / s),   F = -c - E q
 *   fluid   U = -G (LY - y)^2 / (2 NU) + C (LY - y)
 *
 * F makes U(0) = 0, the fluid form makes U(LY) = 0, and E and C make U and
 * U' continuous at y = D.  Every exponential has a non-positive argument, so
 * the formula stays finite however thin the layer is compared with s.
 */
static double layer_profile(double y)
{
    const double depth = LAYER_FRACTION * (double)LY;
    const double gap = (double)LY - depth;
    const double s = sqrt(permeability);
    const double c = body_force * permeability / (double)NU;
    const double q = exp(-depth / s);
    const double ratio = gap / s;
    const double e =
        (body_force * gap * gap / (2.0 * (double)NU) -
         c * (1.0 - q) - ratio * c * q) /
        ((1.0 - q * q) + ratio * (1.0 + q * q));

    if (y <= depth) {
        const double f = -c - e * q;
        return c + e * exp(-(depth - y) / s) + f * exp(-y / s);
    }

    const double coefficient =
        body_force * gap / (double)NU - (e * (1.0 + q * q) + c * q) / s;
    const double distance = (double)LY - y;

    return -body_force * distance * distance / (2.0 * (double)NU) +
           coefficient * distance;
}

static double exact_profile(double y)
{
    if (y < 0.0 || y > (double)LY) {
        return 0.0;
    }
    if (layer_case) {
        return layer_profile(y);
    }
    return body_force * permeability / (double)NU * uniform_shape(y);
}

static Real channel_velocity(Real x, Real y, Real z, Real t, int component)
{
    (void)x;
    (void)z;
    (void)t;

    return component == 0 ? (Real)exact_profile((double)y) : (Real)0;
}

static Real channel_forcing(Real x, Real y, Real z, Real t, int component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    return component == 0 ? (Real)body_force : (Real)0;
}

static Real channel_permeability(Real x, Real y, Real z, Real t,
                                 int component)
{
    (void)x;
    (void)z;
    (void)t;
    (void)component;

    if (layer_case && (double)y > LAYER_FRACTION * (double)LY) {
        return (Real)FREE_FLUID_PERMEABILITY;
    }
    return (Real)permeability;
}

static Real zero_pressure(Real x, Real y, Real z, Real t)
{
    (void)x;
    (void)y;
    (void)z;
    (void)t;

    return (Real)0;
}

typedef struct ChannelErrors {
    double relative_l2;
    double linf_x;
    double linf_yz;
    double porous_max_numerical;
    double porous_max_exact;
} ChannelErrors;

/* Cells on an X or Z face hold the exact profile and are left out. */
static ChannelErrors measure(const Decomp *d, const SolverMemState *state)
{
    double error_sum = 0.0;
    double exact_sum = 0.0;
    double linf_x = 0.0;
    double linf_yz = 0.0;
    double porous_numerical = 0.0;
    double porous_exact = 0.0;
    const double depth = LAYER_FRACTION * (double)LY;

    for (int k = 0; k < d->n[2]; k++) {
        const int gk = decomp_global(d, k, 2);
        if (gk == 0 || gk == d->n_global[2] - 1) {
            continue;
        }
        for (int j = 0; j < d->n[1]; j++) {
            const int gj = decomp_global(d, j, 1);
            const double y = (double)centered_physical_coord(gj, 1);
            const double exact = exact_profile(y);
            const size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                const int gi = decomp_global(d, i, 0);
                if (gi == 0 || gi == d->n_global[0] - 1) {
                    continue;
                }
                const size_t at = row + (size_t)i;
                const double ux = (double)state->u.v_x[at];
                const double uy = (double)state->u.v_y[at];
                const double uz = (double)state->u.v_z[at];
                const double difference = fabs(ux - exact);

                error_sum += difference * difference + uy * uy + uz * uz;
                exact_sum += exact * exact;
                if (difference > linf_x) {
                    linf_x = difference;
                }
                if (fabs(uy) > linf_yz) {
                    linf_yz = fabs(uy);
                }
                if (fabs(uz) > linf_yz) {
                    linf_yz = fabs(uz);
                }
                if (y <= depth) {
                    if (fabs(ux) > porous_numerical) {
                        porous_numerical = fabs(ux);
                    }
                    if (fabs(exact) > porous_exact) {
                        porous_exact = fabs(exact);
                    }
                }
            }
        }
    }

    const ChannelErrors errors = {
        .relative_l2 = sqrt((double)par_sum_real((Real)error_sum) /
                            (double)par_sum_real((Real)exact_sum)),
        .linf_x = (double)par_max_real((Real)linf_x),
        .linf_yz = (double)par_max_real((Real)linf_yz),
        .porous_max_numerical = (double)par_max_real((Real)porous_numerical),
        .porous_max_exact = (double)par_max_real((Real)porous_exact),
    };
    return errors;
}

static void usage(const char *program)
{
    if (par_rank() == 0) {
        fprintf(stderr,
                "usage: %s [uniform|layer] [K] [config-file]\n"
                "K must lie in [1e-6, 1e2]; the default is 1e-2.\n",
                program);
    }
    par_finalize();
    exit(2);
}

int main(int argc, char **argv)
{
    par_init(&argc, &argv);

    if (argc > 4) {
        usage(argv[0]);
    }
    if (argc > 1) {
        if (strcmp(argv[1], "layer") == 0) {
            layer_case = 1;
        } else if (strcmp(argv[1], "uniform") != 0) {
            usage(argv[0]);
        }
    }
    if (argc > 2) {
        char *end = NULL;
        permeability = strtod(argv[2], &end);
        /* Past these bounds it is the exact formulas that lose digits, not
         * the solver: better to stop than to compare against a wrong
         * reference. */
        if (end == argv[2] || *end != '\0' ||
            !(permeability >= 1e-6 && permeability <= 1e2)) {
            usage(argv[0]);
        }
    }
    if (argc > 3) {
        params_load(argv[3]);
    }
    if (sim.width < 3 || sim.depth < 3) {
        if (par_rank() == 0) {
            fprintf(stderr, "brinkman_channel: width and depth must be at "
                    "least 3, or no cell is left to measure\n");
        }
        par_finalize();
        return 2;
    }

    /* The force that brings the maximum velocity to about one: exactly one on
     * the centreline for uniform K, and in the free gap when the layer is
     * nearly solid. */
    if (layer_case) {
        const double gap = (1.0 - LAYER_FRACTION) * (double)LY;
        body_force = 8.0 * (double)NU / (gap * gap);
    } else {
        body_force = (double)NU /
                     (permeability * uniform_shape(0.5 * (double)LY));
    }

    Data data = {
        .name = layer_case ? "Brinkman channel, porous layer"
                           : "Brinkman channel, uniform K",
        .bc_velocity = channel_velocity,
        .forcing_fn = channel_forcing,
        .porosity_fn = channel_permeability,
        .porosity_time_dependent = 0,
        .velocity_fn = channel_velocity,
        .pressure_fn = zero_pressure,
    };
    SolverMemState state;
    SolverStats stats = {0};
    Decomp decomp;

    const int process_grid[3] = {0, 0, 0};
    par_topology_init(process_grid);
    decomp_init_mpi(&decomp);
    solver_init(&decomp, &state, &data, NULL);
    solver_solve(&decomp, &state, &data, &stats, 0);

    const Real velocity_time = (Real)STEPS * (Real)DT;
    const SolverErrorNorms norms =
        compute_solver_error_norms(&decomp, &state, &data, velocity_time,
                                   velocity_time - (Real)DT / (Real)2);
    const ChannelErrors errors = measure(&decomp, &state);
    const int failed = !(errors.relative_l2 <= MAX_RELATIVE_ERROR);

    print_solver_error_norms(&decomp, &norms, velocity_time,
                             velocity_time - (Real)DT / (Real)2);
    if (par_rank() == 0) {
        printf("\nBrinkman channel (%s):\n",
               layer_case ? "layer" : "uniform");
        printf("  K: %.3e\n", permeability);
        printf("  sqrt(K) / dy: %.4f\n", sqrt(permeability) / (double)DY);
        printf("  dy: %.10e\n", (double)DY);
        printf("  body force G: %.10e\n", body_force);
        printf("  relative L2 error u: %.10e\n", errors.relative_l2);
        printf("  Linf error u_x: %.10e\n", errors.linf_x);
        printf("  Linf u_y, u_z: %.10e\n", errors.linf_yz);
        if (layer_case) {
            printf("  porous max |u_x|: numerical %.10e, exact %.10e\n",
                   errors.porous_max_numerical, errors.porous_max_exact);
        }
        printf("  threshold: relative L2 <= %.2e  %s\n",
               MAX_RELATIVE_ERROR, failed ? "FAILED" : "ok");
    }

    par_finalize();
    return failed ? 1 : 0;
}
