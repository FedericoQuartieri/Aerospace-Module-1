#include "init.h"
#include "utils.h"

int y_slice_dim = 4; // Number of SIMD vectors treated as one cache-friendly block in the y-direction.
int z_slice_dim = 8;


bool init_temporary_arrays(DTYPE **tmp_dxx, DTYPE **rhs_dxx, DTYPE **u_dxx,
    DTYPE **simd_tmp_dyy, DTYPE **simd_update_dyy, DTYPE **bc_right_dyy, DTYPE **scalar_w_dyy,
    DTYPE **simd_tmp_dzz, DTYPE **simd_update_dzz, DTYPE **bc_right_dzz, DTYPE **scalar_w_dzz) {

    *tmp_dxx = NULL;
    *rhs_dxx = NULL;
    *u_dxx = NULL;
    *simd_tmp_dyy = NULL;
    *simd_update_dyy = NULL;
    *bc_right_dyy = NULL;
    *scalar_w_dyy = NULL;
    *simd_tmp_dzz = NULL;
    *simd_update_dzz = NULL;
    *bc_right_dzz = NULL;
    *scalar_w_dzz = NULL;

    /* Dxx tridiagonal system */
    *tmp_dxx = (DTYPE *)xmalloc(WIDTH * sizeof(DTYPE));
    *rhs_dxx = (DTYPE *)xmalloc(WIDTH * sizeof(DTYPE));
    *u_dxx = (DTYPE *)xmalloc(WIDTH * sizeof(DTYPE));

    /* Dyy tridiagonal system */
    *simd_tmp_dyy = (DTYPE *)xmalloc(HEIGHT * sizeof(DTYPE) * y_slice_dim * VLEN);
    *simd_update_dyy = (DTYPE *)xmalloc(HEIGHT * sizeof(DTYPE) * y_slice_dim * VLEN);
    *bc_right_dyy = (DTYPE *)xmalloc(y_slice_dim * VLEN * sizeof(DTYPE));
    *scalar_w_dyy = (DTYPE *)xmalloc(HEIGHT * sizeof(DTYPE));

    /* Dzz tridiagonal system */
    *simd_tmp_dzz = (DTYPE *)xmalloc(DEPTH * sizeof(DTYPE) * z_slice_dim * VLEN);
    *simd_update_dzz = (DTYPE *)xmalloc(DEPTH * sizeof(DTYPE) * z_slice_dim * VLEN);
    *bc_right_dzz = (DTYPE *)xmalloc(z_slice_dim * VLEN * sizeof(DTYPE));
    *scalar_w_dzz = (DTYPE *)xmalloc(DEPTH * sizeof(DTYPE));

    if (!*tmp_dxx || !*rhs_dxx || !*u_dxx ||
        !*simd_tmp_dyy || !*simd_update_dyy || !*bc_right_dyy || !*scalar_w_dyy ||
        !*simd_tmp_dzz || !*simd_update_dzz || !*bc_right_dzz || !*scalar_w_dzz) {
        free_temporary_arrays(*tmp_dxx, *rhs_dxx, *u_dxx,
            *simd_tmp_dyy, *simd_update_dyy, *bc_right_dyy, *scalar_w_dyy,
            *simd_tmp_dzz, *simd_update_dzz, *bc_right_dzz, *scalar_w_dzz);
        *tmp_dxx = NULL;
        *rhs_dxx = NULL;
        *u_dxx = NULL;
        *simd_tmp_dyy = NULL;
        *simd_update_dyy = NULL;
        *bc_right_dyy = NULL;
        *scalar_w_dyy = NULL;
        *simd_tmp_dzz = NULL;
        *simd_update_dzz = NULL;
        *bc_right_dzz = NULL;
        *scalar_w_dzz = NULL;
        return false;
    }

    return true;
}

void free_temporary_arrays(DTYPE *tmp_dxx, DTYPE *rhs_dxx, DTYPE *u_dxx,
    DTYPE *simd_tmp_dyy, DTYPE *simd_update_dyy, DTYPE *bc_right_dyy, DTYPE *scalar_w_dyy,
    DTYPE *simd_tmp_dzz, DTYPE *simd_update_dzz, DTYPE *bc_right_dzz, DTYPE *scalar_w_dzz) {

    free(tmp_dxx);
    free(rhs_dxx);
    free(u_dxx);
    free(simd_tmp_dyy);
    free(simd_update_dyy);
    free(bc_right_dyy);
    free(scalar_w_dyy);
    free(simd_tmp_dzz);
    free(simd_update_dzz);
    free(bc_right_dzz);
    free(scalar_w_dzz);
}
