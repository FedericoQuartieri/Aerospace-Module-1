#ifndef EQUATIONS_H
#define EQUATIONS_H

#include "ftype.h"

void compute_w(const ftype *__restrict__ k,
               uint32_t depth,
               uint32_t height,
               uint32_t width,
               ftype nu,
               ftype dt,
               ftype *__restrict__ w);

void compute_wDxx_rhs(const ftype *__restrict__ k, /* Porosity. */
                      /* Pressure from previous half-step. */
                      const ftype *__restrict__ p,
                      /* Pressure correction from previous half-step. */
                      const ftype *__restrict__ phi,
                      /* (I - wDxx) velocity from previous step */
                      const ftype *__restrict__ eta_x,
                      const ftype *__restrict__ eta_y,
                      const ftype *__restrict__ eta_z,
                      /* (I - wDyy) velocity from previous step */
                      const ftype *__restrict__ zeta_x,
                      const ftype *__restrict__ zeta_y,
                      const ftype *__restrict__ zeta_z,
                      /* Velocity from previous step */
                      const ftype *__restrict__ u_x,
                      const ftype *__restrict__ u_y,
                      const ftype *__restrict__ u_z,
                      uint32_t depth,
                      uint32_t height,
                      uint32_t width,
                      ftype nu, /* Viscosity */
                      ftype dt, /* Timestep size */
                      ftype dx, /* Grid cell size */
                      ftype *__restrict__ rhs_x,
                      ftype *__restrict__ rhs_y,
                      ftype *__restrict__ rhs_z);

void solve_momentum(const ftype *__restrict__ k, /* Porosity. */
                    const ftype *__restrict__ w,
                    uint32_t depth,
                    uint32_t height,
                    uint32_t width,
                    ftype nu,
                    ftype dt,
                    ftype dx,
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
