#ifndef VELOCITY_FIELD_H
#define VELOCITY_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"
#include "data.h"

typedef struct {
    DTYPE *v_x;
    DTYPE *v_y;
    DTYPE *v_z;
} VelocityField;

void initialize_velocity_field(VelocityField *v_field);
void free_velocity_field(VelocityField *v_field);
void rand_fill_velocity_field(VelocityField *v_field);
DTYPE compute_velocity_x_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_y_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_z_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_xx_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_yy_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
DTYPE compute_velocity_zz_grad(DTYPE *v_component, size_t i, size_t j, size_t k);
void update_delta_left_velocity_boundary(VelocityField *v_field, int time_step, const Data *data);
void update_delta_right_velocity_boundary(VelocityField *v_field, int time_step, const Data *data);
//void get_boundary_velocity(size_t i, size_t j, size_t k, DTYPE t, const Data *data, DTYPE *vx, DTYPE *vy, DTYPE *vz);
DTYPE get_boundary_velocity(int i, int j, int k, int time_step, const Data *data,
                            int v_component);
#endif // VELOCITY_FIELD_H