#ifndef INIT_H
#define INIT_H

#include "constants.h"
#include "../optimization/neon_instructions.h"

#include <stdbool.h>
#include <stdlib.h>

bool init_temporary_arrays(DTYPE **tmp_dxx, DTYPE **rhs_dxx, DTYPE **u_dxx,
    DTYPE **simd_tmp_dyy, DTYPE **simd_update_dyy, DTYPE **bc_right_dyy, DTYPE **scalar_w_dyy,
    DTYPE **simd_tmp_dzz, DTYPE **simd_update_dzz, DTYPE **bc_right_dzz, DTYPE **scalar_w_dzz);

void free_temporary_arrays(DTYPE *tmp_dxx, DTYPE *rhs_dxx, DTYPE *u_dxx,
    DTYPE *simd_tmp_dyy, DTYPE *simd_update_dyy, DTYPE *bc_right_dyy, DTYPE *scalar_w_dyy,
    DTYPE *simd_tmp_dzz, DTYPE *simd_update_dzz, DTYPE *bc_right_dzz, DTYPE *scalar_w_dzz);

    
#endif
