#ifndef G_FIELD_H
#define G_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"
#include "pressure.h"
#include "velocity_field.h"
#include "force_field.h"
#include "data.h"

typedef struct {
    DTYPE *g_x;
    DTYPE *g_y;
    DTYPE *g_z;
} GField;

void initialize_g_field(GField *g_field);
void free_g_field(GField *g_field);
void compute_g(GField *g_field, Pressure *pressure, DTYPE *k, VelocityField *Eta, VelocityField *Zeta, VelocityField *U, int time_step, const Data *data);
DTYPE g_value(size_t i, size_t j, size_t k,
              DTYPE *pressure_star,
              DTYPE k_i,
              DTYPE *Eta_prev,
              DTYPE *Zeta_prev,
              DTYPE *U_prev,
              int time_step,
              const Data *data,
              int v_component);
            
#endif // G_FIELD_H