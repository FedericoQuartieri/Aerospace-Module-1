#include "force_field.h"
#include <stdlib.h>
#include <string.h>

void initialize_force_field(ForceField *f_field) {
    f_field->f_x = (DTYPE*) malloc(GRID_SIZE);
    f_field->f_y = (DTYPE*) malloc(GRID_SIZE);
    f_field->f_z = (DTYPE*) malloc(GRID_SIZE);
}

void rand_fill_force_field(ForceField *f_field) {
    rand_fill(f_field->f_x);
    rand_fill(f_field->f_y);
    rand_fill(f_field->f_z);
}

void free_force_field(ForceField *f_field) {
    free(f_field->f_x);
    free(f_field->f_y);
    free(f_field->f_z);
}