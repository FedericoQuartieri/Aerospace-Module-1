#include "solver.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    Solver solver = {0};
    FILE *stream = tmpfile();
    char report[2048];
    size_t bytes;

    if (stream == NULL) {
        return 1;
    }
    solver.grid.cell_count = 8;
    solver.stats.completed_steps = 4;
    solver.stats.momentum_kernel_ns[DIRECTION_X] = 320;
    solver.stats.pressure_kernel_ns[DIRECTION_X] = 160;
    solver.stats.timestep_compute_ns = 800;

    solver_print_stats(&solver, stream);
    if (fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return 1;
    }
    bytes = fread(report, 1, sizeof(report) - 1, stream);
    fclose(stream);
    report[bytes] = '\0';

    return strstr(report,
                  "Momentum X per cell: 10.000000 ns/(step cell)") == NULL ||
           strstr(report,
                  "Pressure X per cell: 5.000000 ns/(step cell)") == NULL ||
           strstr(report,
                  "Timestep compute per cell: "
                  "0.025000 us/(step cell)") == NULL ||
           strstr(report,
                  "Timestep compute per cell: "
                  "25.000000 ns/(step cell)") == NULL;
}
