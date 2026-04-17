#include "test_common.h"
#include "../include/tridiagonal_blocks.h"

/*
 * Unit tests for the Thomas algorithm implementations.
 *
 * The solver takes a tridiagonal system of the form:
 *
 *   [ 1+2w_0   w_0     0    0   ] [u_0]   [f_0]
 *   [  w_1   1+2w_1   w_1   0   ] [u_1] = [f_1]
 *   [   0     w_2   1+2w_2  w_2 ] [u_2]   [f_2]
 *   [   ...                     ] [...]   [...]
 *
 * where w_i = -gamma * DX^-2 < 0 (off-diagonal coefficients).
 *
 * Thomas_Same_Direction: Dirichlet BCs at both ends (u[0] and u[n-1] preset).
 * Thomas_Pressure:       Neumann BC at left, modified Neumann at right.
 */

#define N_SMALL 8       /* small 1D system size for unit tests */
#define TOL     1e-10   /* tolerance for double precision */

/* ------------------------------------------------------------------ */
/* Helper: allocate and fill a 1D array of size n with a constant value */
static DTYPE *alloc_const(unsigned int n, DTYPE val) {
    DTYPE *arr = (DTYPE *) malloc(n * sizeof(DTYPE));
    for (unsigned int i = 0; i < n; i++) arr[i] = val;
    return arr;
}

/* ------------------------------------------------------------------ */
/* Test 1 – Thomas_Same_Direction: constant solution u[i] = 1
 *
 * For u[i] = C = 1 the interior RHS is:
 *   f[i] = (1+2w)*C - w*C - w*C = C = 1
 * with u[0] = u[n-1] = 1 as Dirichlet BCs.
 */
int test_thomas_same_constant(void) {
    printf("\n====== TEST: Thomas_Same_Direction – constant solution ======\n");

    const unsigned int n = N_SMALL;
    const DTYPE w_val = -0.1;   /* gamma coefficient (< 0) */
    const DTYPE C = 1.0;        /* constant solution */

    DTYPE *w   = alloc_const(n, w_val);
    DTYPE *rhs = alloc_const(n, C);       /* f[i] = C for all i */
    DTYPE *tmp = alloc_const(n, 0.0);
    DTYPE *u   = alloc_const(n, 0.0);

    /* Dirichlet boundary values */
    u[0]   = C;
    u[n-1] = C;

    bool same_direction = true;  
    Thomas_Algorithm(w, n, tmp, rhs, u, same_direction);

    int failed = 0;
    for (unsigned int i = 0; i < n; i++) {
        if (fabs(u[i] - C) > TOL) {
            fprintf(stderr, "  u[%u] = %e, expected %e, err = %e\n",
                    i, u[i], C, fabs(u[i] - C));
            failed++;
        }
    }

    free(w); free(rhs); free(tmp); free(u);

    if (failed) {
        printf("FAIL: %d elements wrong.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: constant solution recovered (tol=%e).\n", TOL);
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 2 – Thomas_Same_Direction: linear solution u[i] = i
 *
 * A linear sequence satisfies (u[i-1] - 2u[i] + u[i+1]) = 0, so:
 *   f[i] = (1+2w)*i - w*(i-1) - w*(i+1) = i   for interior points.
 * Dirichlet BCs: u[0]=0, u[n-1]=n-1.
 */
int test_thomas_same_linear(void) {
    printf("\n====== TEST: Thomas_Same_Direction – linear solution ======\n");

    const unsigned int n = N_SMALL;
    const DTYPE w_val = -0.1;

    DTYPE *w   = alloc_const(n, w_val);
    DTYPE *rhs = (DTYPE *) malloc(n * sizeof(DTYPE));
    DTYPE *tmp = alloc_const(n, 0.0);
    DTYPE *u   = alloc_const(n, 0.0);

    /* RHS: f[i] = i (interior) */
    for (unsigned int i = 0; i < n; i++) rhs[i] = (DTYPE) i;

    /* Dirichlet BCs */
    u[0]   = 0.0;
    u[n-1] = (DTYPE)(n - 1);

    bool same_direction = true;
    Thomas_Algorithm(w, n, tmp, rhs, u, same_direction);

    int failed = 0;
    for (unsigned int i = 0; i < n; i++) {
        DTYPE expected = (DTYPE) i;
        if (fabs(u[i] - expected) > TOL) {
            fprintf(stderr, "  u[%u] = %e, expected %e, err = %e\n",
                    i, u[i], expected, fabs(u[i] - expected));
            failed++;
        }
    }

    free(w); free(rhs); free(tmp); free(u);

    if (failed) {
        printf("FAIL: %d elements wrong.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: linear solution recovered (tol=%e).\n", TOL);
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 3 – Thomas_Pressure: constant solution u[i] = 1
 *
 * Thomas_Pressure uses a scalar w and Neumann BCs. For the system
 * implemented in the code, a constant solution u[i] = C produces:
 *
 *   f[0]   = (1-2w)*C - 2w*C       = C(1 - 4w)
 *   f[i]   = -w*C + (1-2w)*C - w*C = C          (interior)
 *   f[n-1] = -w*C + (1-w)*C        = C(1 - 2w)
 *
 * With C=1 and w=-0.1 (so 1-2w=1.2, 1-4w=1.4, 1-w=1.1, 1-2w=1.2):
 *   f = {1.4, 1.2, 1.2, 1.2, 1.2} for n=5
 *
 * We verify that Thomas_Pressure recovers u = {1,1,1,1,1}.
 */
int test_thomas_pressure_constant(void) {
    printf("\n====== TEST: Thomas_Pressure – constant solution ======\n");

    const unsigned int n = 5;
    const DTYPE w  = -0.1;
    const DTYPE C  = 1.0;

    /*
     * Build the RHS analytically for u = C:
     *   f[0]   = C * (1 - 4*w)   (left Neumann row)
     *   f[i]   = C                (interior rows)
     *   f[n-1] = C * (1 - 2*w)   (right Neumann row, diagonal = 1-w)
     */
    DTYPE *rhs = (DTYPE *) malloc(n * sizeof(DTYPE));
    DTYPE *tmp = alloc_const(n, 0.0);
    DTYPE *u   = alloc_const(n, 0.0);

    rhs[0]   = C * (1.0 - 4.0 * w);   /* = 1.4 */
    for (unsigned int i = 1; i < n - 1; i++) rhs[i] = C;
    rhs[n-1] = C * (1.0 - 2.0 * w);   /* = 1.2 */

    Thomas_Pressure(w, n, tmp, rhs, u);

    int failed = 0;
    for (unsigned int i = 0; i < n; i++) {
        if (fabs(u[i] - C) > TOL) {
            fprintf(stderr, "  u[%u] = %e, expected %e, err = %e\n",
                    i, u[i], C, fabs(u[i] - C));
            failed++;
        }
    }

    free(rhs); free(tmp); free(u);

    if (failed) {
        printf("FAIL: %d elements wrong.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: constant pressure solution recovered (tol=%e).\n", TOL);
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 4 – Thomas_Same_Direction: null RHS with zero boundary
 *
 * With u[0]=0, u[n-1]=0 and f[i]=0 (interior), the only solution is
 * the trivial one u[i]=0 for all i.
 */
int test_thomas_same_zero(void) {
    printf("\n====== TEST: Thomas_Same_Direction – zero solution ======\n");

    const unsigned int n = N_SMALL;
    const DTYPE w_val = -0.15;

    DTYPE *w   = alloc_const(n, w_val);
    DTYPE *rhs = alloc_const(n, 0.0);
    DTYPE *tmp = alloc_const(n, 0.0);
    DTYPE *u   = alloc_const(n, 0.0);

    u[0]   = 0.0;
    u[n-1] = 0.0;

    bool same_direction = true;
    Thomas_Algorithm(w, n, tmp, rhs, u, same_direction);

    int failed = 0;
    for (unsigned int i = 0; i < n; i++) {
        if (fabs(u[i]) > TOL) {
            fprintf(stderr, "  u[%u] = %e, expected 0.0\n", i, u[i]);
            failed++;
        }
    }

    free(w); free(rhs); free(tmp); free(u);

    if (failed) {
        printf("FAIL: %d elements non-zero.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: zero solution recovered (tol=%e).\n", TOL);
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */

int main(void) {
    printf("========== Thomas Algorithm Tests ==========\n\n");

    int result = TEST_PASS;

    if (test_thomas_same_constant()        != TEST_PASS) result = TEST_FAIL;
    if (test_thomas_same_linear()          != TEST_PASS) result = TEST_FAIL;
    if (test_thomas_same_zero()            != TEST_PASS) result = TEST_FAIL;
    if (test_thomas_pressure_constant()    != TEST_PASS) result = TEST_FAIL;

    printf("\n============================================\n");
    printf(result == TEST_PASS ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("============================================\n");

    return result;
}
