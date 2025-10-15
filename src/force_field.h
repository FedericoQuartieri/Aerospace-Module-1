#ifndef FORCE_FIELD_H
#define FORCE_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"

typedef struct {
    DTYPE *f_x;
    DTYPE *f_y;
    DTYPE *f_z;
} ForceField;

void initialize_force_field(ForceField *f_field);
void free_force_field(ForceField *f_field);

#endif // FORCE_FIELD_H