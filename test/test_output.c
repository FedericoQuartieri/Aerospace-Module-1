#include "solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static Real zero_vector(Real x,
                        Real y,
                        Real z,
                        Real time,
                        Direction component)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    (void)component;
    return (Real)0;
}

static Real zero_scalar(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)0;
}

static Real unit_scalar(Real x, Real y, Real z, Real time)
{
    (void)x;
    (void)y;
    (void)z;
    (void)time;
    return (Real)1;
}

int main(void)
{
    static const ProblemDefinition problem = {
        "output", zero_vector, zero_scalar, zero_vector,
        zero_vector, unit_scalar
    };
    SolverConfig config = solver_default_config();
    Solver solver = {0};
    FILE *file;
    char header[4096];
    size_t bytes;
    int result = 1;

    config.extent[DIRECTION_X] = 4;
    config.extent[DIRECTION_Y] = 4;
    config.extent[DIRECTION_Z] = 4;
    config.dt = (Real)0.01;
    config.steps = 2;
    config.output_frequency = 2;
    config.output_directory = "test-output";
    if (solver_init(&solver, &config, &problem) != SOLVER_SUCCESS ||
        solver_solve(&solver) != SOLVER_SUCCESS) {
        goto cleanup;
    }
    file = fopen("test-output/solution_000000.vti", "rb");
    if (file == NULL) goto cleanup;
    bytes = fread(header, 1, sizeof(header) - 1, file);
    fclose(file);
    header[bytes] = '\0';
    if (strstr(header,
               "Name=\"TimeValue\" NumberOfTuples=\"1\" "
               "format=\"ascii\">0</DataArray>") == NULL ||
        strstr(header,
               "Name=\"PressureTime\" NumberOfTuples=\"1\" "
               "format=\"ascii\">0</DataArray>") == NULL ||
        strstr(header, "Name=\"Velocity\"") == NULL ||
        strstr(header, "Name=\"Pressure\"") == NULL) {
        goto cleanup;
    }
    file = fopen("test-output/solution_000002.vti", "rb");
    if (file == NULL) goto cleanup;
    fclose(file);
    result = 0;

cleanup:
    solver_destroy(&solver);
    remove("test-output/solution_000000.vti");
    remove("test-output/solution_000002.vti");
    rmdir("test-output");
    if (result == 0) {
        config.output_directory = "test-output-missing-parent/nested";
        if (solver_init(&solver, &config, &problem) != SOLVER_OUTPUT_ERROR) {
            result = 1;
        }
        solver_destroy(&solver);
    }
    return result;
}
