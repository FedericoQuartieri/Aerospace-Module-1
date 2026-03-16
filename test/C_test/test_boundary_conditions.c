#include "test_common.h"
#include "../include/function.h"

/*
 * Tests for velocity boundary condition updates.
 *
 * The boundary function used is test_manufactured_Vboundary.txt:
 *   v_x = sin(t) * sin(pi*x) * sin(pi*y) * sin(pi*z)
 *   v_y = sin(t) * cos(pi*x) * cos(pi*y) * cos(pi*z)
 *   v_z = sin(t) * cos(pi*x) * sin(pi*y) * (cos(pi*z) + sin(pi*z))
 *
 * These tests verify properties orthogonal to test_init_velocity.c:
 *   1. Interior points are never overwritten by boundary updates.
 *   2. Boundary values are finite (no NaN / Inf) after both left and right updates.
 *   3. Boundary values vary with time (the update is not static).
 */

#define SENTINEL 3.14159265358979323846   /* value written to interior before update */
#define BC_FILE  "../function_files/test_manufactured_Vboundary.txt"

/* Returns 1 if (i,j,k) is a strictly interior (non-boundary) point */
static int is_interior(int i, int j, int k) {
    return (i > 0 && i < WIDTH  - 1 &&
            j > 0 && j < HEIGHT - 1 &&
            k > 0 && k < DEPTH  - 1);
}

/* Returns 1 if (i,j,k) is on any face/edge/vertex of the grid */
static int is_boundary_pt(int i, int j, int k) {
    return !is_interior(i, j, k);
}

/* ------------------------------------------------------------------ */
/* Test 1 – Interior points are NOT modified by boundary updates
 *
 * Procedure:
 *   - Allocate a VelocityField.
 *   - Write SENTINEL to every interior cell.
 *   - Run both left and right boundary updates.
 *   - Verify all interior cells still hold SENTINEL.
 */
int test_boundary_interior_preserved(void) {
    printf("\n====== TEST: Interior Points Not Modified by Boundary Update ======\n");

    VelocityField v;
    function_handle bc = parse_function(BC_FILE);
    if (!bc) {
        fprintf(stderr, "Error: could not load boundary file %s\n", BC_FILE);
        return TEST_FAIL;
    }

    initialize_velocity_field(&v, bc);

    /* Paint all interior cells with SENTINEL */
    for (int k = 0; k < DEPTH; k++)
        for (int j = 0; j < HEIGHT; j++)
            for (int i = 0; i < WIDTH; i++) {
                if (is_interior(i, j, k)) {
                    size_t idx = rowmaj_idx(i, j, k);
                    v.v_x[idx] = SENTINEL;
                    v.v_y[idx] = SENTINEL;
                    v.v_z[idx] = SENTINEL;
                }
            }

    /* Apply boundary updates (time_step = 5) */
    update_delta_left_velocity_boundary(&v, bc, 5);
    update_delta_right_velocity_boundary(&v, bc, 5);

    /* Check that interior cells still hold SENTINEL */
    int failed = 0;
    for (int k = 1; k < DEPTH - 1; k++)
        for (int j = 1; j < HEIGHT - 1; j++)
            for (int i = 1; i < WIDTH - 1; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                if (v.v_x[idx] != SENTINEL || v.v_y[idx] != SENTINEL || v.v_z[idx] != SENTINEL) {
                    if (failed < 5)
                        fprintf(stderr, "  Interior (%d,%d,%d) modified: vx=%e vy=%e vz=%e\n",
                                i, j, k, v.v_x[idx], v.v_y[idx], v.v_z[idx]);
                    failed++;
                }
            }

    free_velocity_field(&v);
    destroy_function(bc);

    if (failed > 0) {
        printf("FAIL: %d interior cells were overwritten by boundary update.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: All interior cells preserved after boundary update.\n");
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 2 – No NaN / Inf at boundary after left and right updates */
int test_boundary_no_nan(void) {
    printf("\n====== TEST: No NaN or Inf at Boundary Points ======\n");

    VelocityField v;
    function_handle bc = parse_function(BC_FILE);
    if (!bc) {
        fprintf(stderr, "Error: could not load boundary file %s\n", BC_FILE);
        return TEST_FAIL;
    }

    initialize_velocity_field(&v, bc);
    update_delta_left_velocity_boundary(&v, bc, 3);
    update_delta_right_velocity_boundary(&v, bc, 3);

    int failed = 0;
    for (int k = 0; k < DEPTH; k++)
        for (int j = 0; j < HEIGHT; j++)
            for (int i = 0; i < WIDTH; i++) {
                if (!is_boundary_pt(i, j, k)) continue;
                size_t idx = rowmaj_idx(i, j, k);
                if (!isfinite(v.v_x[idx]) || !isfinite(v.v_y[idx]) || !isfinite(v.v_z[idx])) {
                    if (failed < 5)
                        fprintf(stderr, "  Non-finite at (%d,%d,%d): vx=%e vy=%e vz=%e\n",
                                i, j, k, v.v_x[idx], v.v_y[idx], v.v_z[idx]);
                    failed++;
                }
            }

    free_velocity_field(&v);
    destroy_function(bc);

    if (failed > 0) {
        printf("FAIL: %d boundary cells contain NaN or Inf.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: All boundary values are finite.\n");
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 3 – Boundary values change with time
 *
 * Since the boundary is sin(t)*..., values at step 1 and step 20 must differ.
 * We verify that at least one boundary point changed between the two updates.
 */
int test_boundary_time_variation(void) {
    printf("\n====== TEST: Boundary Values Vary with Time ======\n");

    function_handle bc = parse_function(BC_FILE);
    if (!bc) {
        fprintf(stderr, "Error: could not load boundary file %s\n", BC_FILE);
        return TEST_FAIL;
    }

    VelocityField v1, v2;
    initialize_velocity_field(&v1, bc);
    initialize_velocity_field(&v2, bc);

    /* Update at two different timesteps */
    const int step_a = 1;
    const int step_b = 20;

    update_delta_left_velocity_boundary(&v1, bc, step_a);
    update_delta_right_velocity_boundary(&v1, bc, step_a);

    update_delta_left_velocity_boundary(&v2, bc, step_b);
    update_delta_right_velocity_boundary(&v2, bc, step_b);

    /* Count boundary cells where the values differ */
    int changed = 0;
    for (int k = 0; k < DEPTH; k++)
        for (int j = 0; j < HEIGHT; j++)
            for (int i = 0; i < WIDTH; i++) {
                if (!is_boundary_pt(i, j, k)) continue;
                size_t idx = rowmaj_idx(i, j, k);
                if (v1.v_x[idx] != v2.v_x[idx] ||
                    v1.v_y[idx] != v2.v_y[idx] ||
                    v1.v_z[idx] != v2.v_z[idx]) {
                    changed++;
                }
            }

    free_velocity_field(&v1);
    free_velocity_field(&v2);
    destroy_function(bc);

    if (changed == 0) {
        printf("FAIL: Boundary values did not change between step %d and step %d.\n",
               step_a, step_b);
        return TEST_FAIL;
    }
    printf("PASS: %d boundary cells changed between step %d and step %d.\n",
           changed, step_a, step_b);
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */
/* Test 4 – Left and right faces are both covered
 *
 * After updating both left (i=0,j=0,k=0 faces) and right (i=WIDTH-1,
 * j=HEIGHT-1, k=DEPTH-1 faces), a sample of cells on each face must be finite.
 */
int test_boundary_faces_covered(void) {
    printf("\n====== TEST: All Six Boundary Faces Are Updated ======\n");

    function_handle bc = parse_function(BC_FILE);
    if (!bc) {
        fprintf(stderr, "Error: could not load boundary file %s\n", BC_FILE);
        return TEST_FAIL;
    }

    VelocityField v;
    initialize_velocity_field(&v, bc);
    update_delta_left_velocity_boundary(&v, bc, 4);
    update_delta_right_velocity_boundary(&v, bc, 4);

    /* Sample a central point on each face */
    typedef struct { int i, j, k; const char *name; } FacePoint;
    FacePoint faces[] = {
        { 0,          HEIGHT/2, DEPTH/2,  "left  (i=0)" },
        { WIDTH-1,    HEIGHT/2, DEPTH/2,  "right (i=W-1)" },
        { WIDTH/2,    0,        DEPTH/2,  "bottom (j=0)" },
        { WIDTH/2,    HEIGHT-1, DEPTH/2,  "top    (j=H-1)" },
        { WIDTH/2,    HEIGHT/2, 0,        "front  (k=0)" },
        { WIDTH/2,    HEIGHT/2, DEPTH-1,  "back   (k=D-1)" },
    };

    int n_faces = (int)(sizeof(faces) / sizeof(faces[0]));
    int failed = 0;

    for (int f = 0; f < n_faces; f++) {
        int i = faces[f].i, j = faces[f].j, k = faces[f].k;
        size_t idx = rowmaj_idx(i, j, k);

        if (!isfinite(v.v_x[idx]) || !isfinite(v.v_y[idx]) || !isfinite(v.v_z[idx])) {
            fprintf(stderr, "  Non-finite on %s face at (%d,%d,%d)\n",
                    faces[f].name, i, j, k);
            failed++;
        } else {
            printf("  Face %-20s (%3d,%3d,%3d): vx=%+.4e  vy=%+.4e  vz=%+.4e\n",
                   faces[f].name, i, j, k,
                   v.v_x[idx], v.v_y[idx], v.v_z[idx]);
        }
    }

    free_velocity_field(&v);
    destroy_function(bc);

    if (failed > 0) {
        printf("FAIL: %d faces have non-finite values.\n", failed);
        return TEST_FAIL;
    }
    printf("PASS: All six boundary faces contain finite values.\n");
    return TEST_PASS;
}

/* ------------------------------------------------------------------ */

int main(void) {
    printf("========== Boundary Conditions Tests ==========\n");
    printf("Grid: WIDTH=%d, HEIGHT=%d, DEPTH=%d\n", WIDTH, HEIGHT, DEPTH);
    printf("DX=%.6f  DY=%.6f  DZ=%.6f  DT=%.6f\n", DX, DY, DZ, DT);
    printf("===============================================\n");

    int result = TEST_PASS;

    if (test_boundary_interior_preserved() != TEST_PASS) result = TEST_FAIL;
    if (test_boundary_no_nan()             != TEST_PASS) result = TEST_FAIL;
    if (test_boundary_time_variation()     != TEST_PASS) result = TEST_FAIL;
    if (test_boundary_faces_covered()      != TEST_PASS) result = TEST_FAIL;

    printf("\n===============================================\n");
    printf(result == TEST_PASS ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("===============================================\n");

    return result;
}
