#ifndef PRESSURE_FIELD_H
#define PRESSURE_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "velocity_field.h"
#include "utils.h"

typedef struct {
    DTYPE *p_x;
    DTYPE *v_y;
    DTYPE *v_z;
} PressureField;

void initialize_pressure_field(VelocityField *v_field);
void free_pressure_field(VelocityField *v_field);

#endif // PRESSURE_FIELD_H