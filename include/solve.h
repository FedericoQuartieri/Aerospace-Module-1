#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include "io_thread.h"
#include "utils.h"
#include "g_field.h"
#include "velocity_field.h"
#include "tridiagonal_blocks.h"
#include "momentum_system.h"
#include "pressure.h"
#include "pressure_system.h"
#include "data.h"

void solve (GField g_field, const Data *data, Pressure pressure, DTYPE* K, 
            VelocityField Eta, VelocityField Zeta, VelocityField U, 
            DTYPE* Beta, DTYPE* Gamma, 
            int write_frequency, bool full_output, VelocityField** U_record, Pressure** P_record);