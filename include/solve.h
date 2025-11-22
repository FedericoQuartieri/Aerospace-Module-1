#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "utils.h"
#include "g_field.h"
#include "velocity_field.h"
#include "tridiagonal_blocks.h"
#include "momentum_system.h"
#include "pressure.h"
#include <sys/stat.h>
#include <sys/types.h>

void solve (GField g_field, Pressure pressure, DTYPE* K, 
            VelocityField Eta, VelocityField Zeta, VelocityField U, DTYPE* Beta, 
            DTYPE* Gamma, DTYPE *u_BC_current_direction, DTYPE *u_BC_derivative_second_direction, DTYPE *u_BC_derivative_third_direction, int write_frequency);

