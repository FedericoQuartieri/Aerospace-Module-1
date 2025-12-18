#ifndef VELOCITY_FIELD_H
#define VELOCITY_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"


typedef struct {
    DTYPE *v_x;
    DTYPE *v_y;
    DTYPE *v_z;
} VelocityField;

typedef DTYPE (*VelocityInitFunc)(int i, int j, int k);

void initialize_velocity_field(VelocityField *v_field, VelocityInitFunc vx_func, VelocityInitFunc vy_func, VelocityInitFunc vz_func);
void free_velocity_field(VelocityField *v_field);
DTYPE compute_velocity_x_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_y_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_z_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_xx_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_yy_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_zz_grad(DTYPE *v_component, size_t i, size_t j, size_t k);

#endif // VELOCITY_FIELD_H