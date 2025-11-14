#include "pressure_field.h"
#include <stdlib.h>
#include <string.h>

void initialize_pressure_field(VelocityField *v_field) {
    v_field->v_x = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_y = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_z = (DTYPE*) malloc(GRID_SIZE);

    rand_fill(v_field->v_x);
    rand_fill(v_field->v_y);
    rand_fill(v_field->v_z);
}

void free_pressure_field(VelocityField *v_field) {
    free(v_field->v_x);
    free(v_field->v_y);
    free(v_field->v_z);
}