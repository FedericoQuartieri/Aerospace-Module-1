#ifndef EQUATIONS_H
#define EQUATIONS_H

#include "ftype.h"

void solve_momentum(const ftype *__restrict__ k, /* Porosity. */
                    uint32_t depth,
                    uint32_t height,
                    uint32_t width,
                    ftype *__restrict__ tmp,
                    /* Pressure from previous half-step. */
                    ftype *__restrict__ p,
                    /* Pressure correction from previous half-step. */
                    ftype *__restrict__ phi,
                    /* (I - wDxx) solution from previous step */
                    ftype *__restrict__ eta_x,
                    ftype *__restrict__ eta_y,
                    ftype *__restrict__ eta_z,
                    /* (I - wDyy) solution from previous step */
                    ftype *__restrict__ zeta_x,
                    ftype *__restrict__ zeta_y,
                    ftype *__restrict__ zeta_z,
                    /* Velocity from previous step */
                    ftype *__restrict__ u_x,
                    ftype *__restrict__ u_y,
                    ftype *__restrict__ u_z);

void solve_pressure(void);

#endif
