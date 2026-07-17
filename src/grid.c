#include "solver.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

bool grid_init(Grid *grid, const SolverConfig *config)
{
    size_t xy;
    Direction direction;

    if (grid == NULL || config == NULL) {
        return false;
    }
    memset(grid, 0, sizeof(*grid));
    for (direction = DIRECTION_X;
         direction < DIRECTION_COUNT;
         direction = (Direction)(direction + 1)) {
        if (config->extent[direction] < 2 ||
            config->extent[direction] > (SIZE_MAX - 1) / 2 ||
            !isfinite(config->domain_length[direction]) ||
            config->domain_length[direction] <= (Real)0) {
            return false;
        }
        grid->extent[direction] = config->extent[direction];
        grid->length[direction] = config->domain_length[direction];
        /* Pressure nodes start at zero; the matching velocity component is
         * shifted by h/2, hence the baseline 2L/(2N-1) metric. */
        grid->spacing[direction] =
            ((Real)2 * grid->length[direction]) /
            (Real)(2 * grid->extent[direction] - 1);
        grid->inverse_spacing[direction] =
            (Real)1 / grid->spacing[direction];
        grid->inverse_spacing_square[direction] =
            grid->inverse_spacing[direction] *
            grid->inverse_spacing[direction];
    }

    if (grid->extent[DIRECTION_X] >
        SIZE_MAX / grid->extent[DIRECTION_Y]) {
        return false;
    }
    xy = grid->extent[DIRECTION_X] * grid->extent[DIRECTION_Y];
    if (xy > SIZE_MAX / grid->extent[DIRECTION_Z]) {
        return false;
    }
    grid->stride[DIRECTION_X] = 1;
    grid->stride[DIRECTION_Y] = grid->extent[DIRECTION_X];
    grid->stride[DIRECTION_Z] = xy;
    grid->cell_count = xy * grid->extent[DIRECTION_Z];
    return true;
}

Real grid_pressure_coordinate(const Grid *grid,
                              Direction direction,
                              size_t index)
{
    return (Real)index * grid->spacing[direction];
}

Real grid_velocity_coordinate(const Grid *grid,
                              Direction coordinate_direction,
                              Direction velocity_component,
                              size_t index)
{
    Real coordinate =
        grid_pressure_coordinate(grid, coordinate_direction, index);
    /* Only the coordinate normal to a component is staggered. */
    if (coordinate_direction == velocity_component) {
        coordinate += grid->spacing[coordinate_direction] / (Real)2;
    }
    return coordinate;
}
