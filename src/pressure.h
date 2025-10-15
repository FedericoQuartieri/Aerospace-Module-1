#ifndef PRESSURE_H
#define PRESSURE_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"


typedef struct {
    DTYPE *p;
} Pressure;

void initialize_pressure(Pressure *pressure);
void free_pressure(Pressure *pressure);
DTYPE compute_pressure_x_grad(DTYPE *p, size_t i, size_t j, size_t k);
DTYPE compute_pressure_y_grad(DTYPE *p, size_t i, size_t j, size_t k);
DTYPE compute_pressure_z_grad(DTYPE *p, size_t i, size_t j, size_t k);

#endif // PRESSURE_H