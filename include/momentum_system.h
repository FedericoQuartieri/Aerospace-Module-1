#ifndef MOMENTUM_SYSTEM_H
#define MOMENTUM_SYSTEM_H

#include <stddef.h>
#include "utils.h"
#include "g_field.h"
#include "velocity_field.h"
#include "tridiagonal_blocks.h"

void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           Pressure *pressure_star,
                           VelocityField Xi,
                           GField g_field,
                           VelocityField Delta,
                           ForceField rhs,
                           DTYPE *Beta,
                           DTYPE *Gamma,
                           const Data *data,
                           int timestep,
                           uint64_t *solver_time,
                           DTYPE *tmp_dxx, DTYPE *rhs_dxx, DTYPE *u_dxx,
                            DTYPE *simd_tmp_dyy, DTYPE *simd_update_dyy, DTYPE *bc_right_dyy, DTYPE *scalar_w_dyy,
                            DTYPE *simd_tmp_dzz, DTYPE *simd_update_dzz, DTYPE *bc_right_dzz, DTYPE *scalar_w_dzz
                        );

void compute_eta_next(VelocityField Eta, VelocityField Delta, VelocityField Zeta, VelocityField U, Pressure *pressure_star, ForceField rhs, VelocityField Xi,
    DTYPE *Gamma, const Data *data, int timestep, DTYPE *tmp_dxx, DTYPE *rhs_dxx, DTYPE *u_dxx);

void compute_zeta_next(VelocityField Zeta, VelocityField Delta, ForceField rhs, VelocityField Eta,
    DTYPE *Gamma, const Data *data, int timestep, DTYPE *simd_tmp_dyy, DTYPE *simd_update_dyy, DTYPE *bc_right_dyy, DTYPE *scalar_w_dyy);
 
void compute_u_next(VelocityField U, VelocityField Delta, ForceField rhs, VelocityField Zeta,
    DTYPE *Gamma, const Data *data, int timestep, DTYPE *simd_tmp_dzz, DTYPE *simd_update_dzz, DTYPE *bc_right_dzz, DTYPE *scalar_w_dzz);
 
void compute_xi(GField g_field, VelocityField U, VelocityField Xi, DTYPE *Beta);
#endif
