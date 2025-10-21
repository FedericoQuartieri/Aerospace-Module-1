#ifndef MOMENTUM_SYSTEM_H
#define MOMENTUM_SYSTEM_H

#include <stddef.h>
#include "utils.h"
#include "g_field.h"
#include "velocity_field.h"
#include "tridiagonal_blocks.h"

/**
 * U, Eta, Zeta are the solution of time step n
 * U_next, Zeta_next, Eta_next are the solution at step n+1
 *  */ 
void solve_momentum_system(VelocityField U, 
                           VelocityField Eta, 
                           VelocityField Zeta, 
                           VelocityField Xi,
                           GField g_field,
                           DTYPE *K,
                           VelocityField U_next,
                           VelocityField Eta_next,
                           VelocityField Zeta_next);
#endif
