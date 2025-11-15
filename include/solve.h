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

void solve (GField g_field, ForceField f_field, Pressure pressure, DTYPE* K, 
            VelocityField Eta, VelocityField Zeta, VelocityField U, DTYPE* Beta, 
            DTYPE* Gamma, int write_frequency);

