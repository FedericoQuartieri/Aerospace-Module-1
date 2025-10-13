#ifndef G_H
#define G_H


#include "defines.h"
#include "helpers.h"

void g(const DTYPE *f_x,
            const DTYPE *f_y,
            const DTYPE *f_z,
            const DTYPE *eta_x,
            const DTYPE *eta_y,
            const DTYPE *eta_z,
            const DTYPE *zeta_x,
            const DTYPE *zeta_y,
            const DTYPE *zeta_z,
            const DTYPE *speed_x,
            const DTYPE *speed_y,
            const DTYPE *speed_z,
            const DTYPE *grad_x,
            const DTYPE *grad_y,
            const DTYPE *grad_z,
            DTYPE *g_x,
            DTYPE *g_y,
            DTYPE *g_z,
            const DTYPE *k_values,
            int depth,
            int height,
            int width);


#endif// G_H    