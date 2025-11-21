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

void fill_with_function(ForceField *f_field, DTYPE (*func)(size_t, size_t, size_t)) {
    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                size_t idx = rowmaj_idx(i,j,k);
                f_field->f_x[idx] = func(i, j, k);
                f_field->f_y[idx] = func(i, j, k);
                f_field->f_z[idx] = func(i, j, k);
            }
        }
    }

}

void free_force_field(ForceField *f_field) {
    free(f_field->f_x);
    free(f_field->f_y);
    free(f_field->f_z);
}