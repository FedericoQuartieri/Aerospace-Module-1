#ifndef G_FIELD_H
#define G_FIELD_H

#include <stddef.h>
#include "../include/constants.h"
#include "utils.h"
#include "pressure.h"
#include "velocity_field.h"
#include "force_field.h"
#include "forcing_parser.h"

typedef struct {
    DTYPE *g_x;
    DTYPE *g_y;
    DTYPE *g_z;
} GField;

void initialize_g_field(GField *g_field);
void free_g_field(GField *g_field);
void compute_g(GField *g_field, forcing_function_t forcing, Pressure *pressure, DTYPE *k, VelocityField *Eta, VelocityField *Zeta, VelocityField *U, int time_step);

#endif // G_FIELD_H