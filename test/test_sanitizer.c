#include "test_support.h"

#include <stdio.h>

int main(void)
{
    SolverConfig config = MANUFACTURED_CASES[0].base_config;
    ErrorReport report;
    config.extent[DIRECTION_X] = 16;
    config.extent[DIRECTION_Y] = 16;
    config.extent[DIRECTION_Z] = 16;
    if (!run_manufactured_case(&MANUFACTURED_CASES[0], &config, &report)) {
        fprintf(stderr, "sanitizer manufactured solve failed\n");
        return 1;
    }
    return 0;
}
