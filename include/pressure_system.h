#ifndef PRESSURE_SYSTEM_H
#define PRESSURE_SYSTEM_H

#include <stddef.h>
#include "utils.h"
#include "pressure.h"
#include "tridiagonal_blocks.h"

/**
 * U, Eta, Zeta are the solution of time step n
 * U_next, Zeta_next, Eta_next are the solution at step n+1
 *  */ 
void solve_pressure_system(VelocityField U_next,
                           Pressure *pressure
                        );

void compute_Psi(VelocityField U_next, Pressure *psi);

void compute_Phi_lower(Pressure *psi, Pressure *phi_lower);
                        
void compute_Phi_higher(Pressure *phi_lower, Pressure *phi_higher);

void compute_pressure(Pressure *phi_higher, Pressure *pressure);
                            
#endif
